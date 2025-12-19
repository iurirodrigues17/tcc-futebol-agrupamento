import sqlite3
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

DB_PATH = "../data/database.sqlite"
SOURCE_TABLE = "team_windows_stats_3_4_5"
TARGET_TABLE = "team_windows_stats_3_4_5_kmeans3_by_league"
METRICS_TABLE = "clustering_stats_3_4_5_by_league_metrics"

USE_FOULS = False      # mean_fouls_for_*, std_fouls_for_*
USE_YELLOWS = True    # mean_yellows_for_*, std_yellows_for_*
USE_REDS = True       # mean_reds_for_*, std_reds_for_*


def main():
    print("DB_PATH:", DB_PATH)
    print("Tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df_all = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Tabela carregada:", df_all.shape)
    print("Colunas de exemplo:", df_all.columns[:20].tolist())

    if "league_id" not in df_all.columns:
        raise ValueError("Coluna 'league_id' não encontrada na tabela de origem.")
    if "result_current" not in df_all.columns:
        raise ValueError("Coluna 'result_current' não encontrada na tabela de origem.")

    leagues = sorted(df_all["league_id"].unique())
    print("Ligas disponíveis:", leagues)

    # Acumuladores
    dfs_out = []
    metrics_rows = []

    # Contexto que NÃO entra como feature
    base_context_cols = [
        "team_id",
        "current_match_id",
        "current_date",
        "league_id",
        "season",
        "opponent_id_current",
        "result_current",
    ]

    for lg in leagues:
        print("\n" + "#" * 80)
        print(f"Liga {lg}")

        df = df_all[df_all["league_id"] == lg].copy()
        n_rows = len(df)
        print("Nº de janelas nesta liga:", n_rows)

        context_cols = [c for c in base_context_cols if c in df.columns]

        # FEATURES: APENAS médias e desvios-padrão, filtrando faltas/cartões via flags
        feature_cols_all = [
            c for c in df.columns
            if (c.startswith("mean_") or c.startswith("std_")) and (c not in context_cols)
        ]

        if len(feature_cols_all) == 0:
            raise ValueError(
                f"Nenhuma feature mean_/std_ encontrada na liga {lg}. "
                "Confira se a tabela possui colunas mean_* e std_*."
            )

        feature_cols = []
        dropped_fouls = []
        dropped_yellows = []
        dropped_reds = []

        for c in feature_cols_all:
            c_low = c.lower()

            if ("fouls_for" in c_low) and (not USE_FOULS):
                dropped_fouls.append(c)
                continue

            if ("yellows_for" in c_low) and (not USE_YELLOWS):
                dropped_yellows.append(c)
                continue

            if ("reds_for" in c_low) and (not USE_REDS):
                dropped_reds.append(c)
                continue

            feature_cols.append(c)

        print(f"Nº de features nesta liga (apenas mean_/std_ selecionadas): {len(feature_cols)}")
        print("Resumo da filtragem nesta liga:")
        print(f"  Usando faltas?    {USE_FOULS}  -> removidas: {len(dropped_fouls)}")
        print(f"  Usando amarelos?  {USE_YELLOWS} -> removidas: {len(dropped_yellows)}")
        print(f"  Usando vermelhos? {USE_REDS}    -> removidas: {len(dropped_reds)}")

        # Segurança contra NaN
        df[feature_cols] = df[feature_cols].fillna(0.0)

        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y_true = df["result_current"].values

        # Métricas com rótulo real
        print("\n=== Rótulo REAL (result_current) nesta liga ===")
        print("Distribuição de rótulos:")
        print(pd.Series(y_true).value_counts())

        try:
            sil_real = silhouette_score(X_scaled, y_true)
            ch_real = calinski_harabasz_score(X_scaled, y_true)
            db_real = davies_bouldin_score(X_scaled, y_true)
        except Exception as e:
            print("Erro ao calcular métricas com rótulo real:", e)
            sil_real = ch_real = db_real = np.nan

        print(f"Silhouette (real): {sil_real:.4f}")
        print(f"Calinski-Harabasz (real): {ch_real:.4f}")
        print(f"Davies-Bouldin (real): {db_real:.4f}")

        # KMeans k=3
        print("\n=== KMeans k=3 nesta liga (sem usar rótulo no treino) ===")
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X_scaled)

        try:
            sil_km = silhouette_score(X_scaled, labels_km)
            ch_km = calinski_harabasz_score(X_scaled, labels_km)
            db_km = davies_bouldin_score(X_scaled, labels_km)
        except Exception as e:
            print("Erro ao calcular métricas internas do KMeans:", e)
            sil_km = ch_km = db_km = np.nan

        ari = adjusted_rand_score(y_true, labels_km)
        nmi = normalized_mutual_info_score(y_true, labels_km)

        print(f"Silhouette (KMeans): {sil_km:.4f}")
        print(f"Calinski-Harabasz (KMeans): {ch_km:.4f}")
        print(f"Davies-Bouldin (KMeans): {db_km:.4f}")
        print(f"ARI (vs rótulo real): {ari:.4f}")
        print(f"NMI (vs rótulo real): {nmi:.4f}")

        print("\nDistribuição de clusters (KMeans k=3):")
        print(pd.Series(labels_km, name="cluster_kmeans_3_stats").value_counts())

        df["cluster_kmeans_3_stats"] = labels_km

        print("\nCrosstab clusters x resultado atual (contagem absoluta):")
        print(pd.crosstab(df["cluster_kmeans_3_stats"], df["result_current"]))

        print("\nCrosstab clusters x resultado atual (proporção por cluster):")
        ct = pd.crosstab(df["cluster_kmeans_3_stats"], df["result_current"], normalize="index")
        print(ct)

        dfs_out.append(df)

        metrics_rows.append({
            "league_id": int(lg),
            "n_janelas": int(n_rows),
            "sil_real": float(sil_real),
            "ch_real": float(ch_real),
            "db_real": float(db_real),
            "sil_kmeans": float(sil_km),
            "ch_kmeans": float(ch_km),
            "db_kmeans": float(db_km),
            "ari": float(ari),
            "nmi": float(nmi),
        })

    # Salvar tabela com clusters
    df_final = pd.concat(dfs_out, ignore_index=True)
    df_final.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
    print(f"\nTabela '{TARGET_TABLE}' salva com todas as janelas + cluster por liga.")

    # Salvar resumo de métricas
    df_metrics = pd.DataFrame(metrics_rows)
    print("\nResumo das métricas por liga (apenas mean_/std_ selecionadas):")
    print(df_metrics)

    df_metrics.to_sql(METRICS_TABLE, conn, if_exists="replace", index=False)
    print(f"\nResumo de métricas salvo na tabela '{METRICS_TABLE}'.")

    conn.close()
    print("Fim do clustering_stats_3_4_5_by_league.py")


if __name__ == "__main__":
    main()
