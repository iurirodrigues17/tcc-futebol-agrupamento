import sqlite3
import pandas as pd
import numpy as np

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

# flags para reuse
USE_FOULS = False        # mean_/std_fouls_...
USE_YELLOWS = True      # mean_/std_yellows_...
USE_REDS = True         # mean_/std_reds_...

TARGET_TABLE_ROWS    = "team_windows_stats_3_4_5_kmeans7_leagues"
TARGET_TABLE_METRICS = "clustering_stats_3_4_5_kmeans7_leagues_metrics"
TARGET_TABLE_DETAILS = "team_windows_stats_3_4_5_kmeans7_leagues_vs_league"


def main():
    print("DB_PATH:", DB_PATH)
    print("Lendo tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Shape da tabela de origem:", df.shape)
    print("Colunas de exemplo:", df.columns[:20].tolist())

    # 1) Seleciona somente colunas mean_/std_
    all_feat_cols = [c for c in df.columns if c.startswith("mean_") or c.startswith("std_")]
    print(f"\nTotal de colunas mean_/std_ encontradas: {len(all_feat_cols)}")

    # 2) Remove grupos dependendo dos flags (faltas / amarelos / vermelhos)
    foul_cols = [c for c in all_feat_cols if "_fouls_" in c or c.endswith("_fouls_for_last5") or c.endswith("_fouls_for_last4") or c.endswith("_fouls_for_last3")]
    yellow_cols = [c for c in all_feat_cols if "_yellows_" in c or c.endswith("_yellows_for_last5") or c.endswith("_yellows_for_last4") or c.endswith("_yellows_for_last3")]
    red_cols = [c for c in all_feat_cols if "_reds_" in c or c.endswith("_reds_for_last5") or c.endswith("_reds_for_last4") or c.endswith("_reds_for_last3")]

    cols_to_remove = []
    if not USE_FOULS:
        cols_to_remove.extend(foul_cols)
    if not USE_YELLOWS:
        cols_to_remove.extend(yellow_cols)
    if not USE_REDS:
        cols_to_remove.extend(red_cols)

    cols_to_remove = sorted(set(cols_to_remove))
    feature_cols = [c for c in all_feat_cols if c not in cols_to_remove]

    print("\nResumo da seleção de features:")
    print(f"  Total original (mean_/std_): {len(all_feat_cols)}")
    print(f"  Usando faltas?      {USE_FOULS}  -> removidas: {len(foul_cols) if not USE_FOULS else 0}")
    print(f"  Usando amarelos?    {USE_YELLOWS} -> removidas: {len(yellow_cols) if not USE_YELLOWS else 0}")
    print(f"  Usando vermelhos?   {USE_REDS}    -> removidas: {len(red_cols) if not USE_REDS else 0}")
    print(f"  Nº de features selecionadas: {len(feature_cols)}")
    print("Exemplos de features selecionadas:", feature_cols[:15])

    # 3) Monta X e padroniza
    X = df[feature_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Shape de X_scaled:", X_scaled.shape)

    # 4) KMeans com k = 7 (uma para cada liga)
    k = 7
    print(f"\n=== KMeans com k={k} (agrupando janelas em 'tipos de liga') ===")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df["cluster_kmeans_7_leagues"] = cluster_labels

    # 5) Métricas internas
    sil = silhouette_score(X_scaled, cluster_labels)
    ch  = calinski_harabasz_score(X_scaled, cluster_labels)
    db  = davies_bouldin_score(X_scaled, cluster_labels)

    print(f"Silhouette (KMeans k={k}): {sil:.4f}")
    print(f"Calinski-Harabasz (KMeans k={k}): {ch:.4f}")
    print(f"Davies-Bouldin (KMeans k={k}): {db:.4f}")

    # 6) Comparação com as ligas (league_id)
    leagues = df["league_id"].values
    ari = adjusted_rand_score(leagues, cluster_labels)
    nmi = normalized_mutual_info_score(leagues, cluster_labels)

    print("\n=== Métricas clusters vs ligas (usando k=7) ===")
    print(f"ARI (clusters vs league_id): {ari:.4f}")
    print(f"NMI (clusters vs league_id): {nmi:.4f}")

    # 7) Distribuições e crosstabs
    print("\nDistribuição geral de ligas (contagem):")
    print(df["league_id"].value_counts())

    print("\nCrosstab cluster x liga (contagem absoluta):")
    ctab_abs = pd.crosstab(df["cluster_kmeans_7_leagues"], df["league_id"])
    print(ctab_abs)

    print("\nCrosstab cluster x liga (proporção por cluster):")
    ctab_row = ctab_abs.div(ctab_abs.sum(axis=1), axis=0)
    print(ctab_row)

    print("\nCrosstab liga x cluster (proporção por liga):")
    ctab_col = pd.crosstab(df["league_id"], df["cluster_kmeans_7_leagues"], normalize="index")
    print(ctab_col)

    # 8) Salvar resultados no banco

    # 8.1. Tabela com todas as janelas + novo cluster k=7
    df.to_sql(TARGET_TABLE_ROWS, conn, if_exists="replace", index=False)
    print(f"\nTabela '{TARGET_TABLE_ROWS}' salva com sucesso no banco.")

    # 8.2. Métricas em formato "1 linha"
    metrics_df = pd.DataFrame([{
        "k": k,
        "n_samples": X_scaled.shape[0],
        "n_features": X_scaled.shape[1],
        "silhouette_kmeans": sil,
        "ch_kmeans": ch,
        "db_kmeans": db,
        "ari_vs_league": ari,
        "nmi_vs_league": nmi,
    }])

    metrics_df.to_sql(TARGET_TABLE_METRICS, conn, if_exists="replace", index=False)
    print(f"Tabela '{TARGET_TABLE_METRICS}' salva com sucesso no banco.")

    # 8.3. Detalhes liga x cluster em formato "long"
    details_df = ctab_col.reset_index().melt(
        id_vars="league_id",
        var_name="cluster_kmeans_7_leagues",
        value_name="proportion_in_league"
    )
    details_df.to_sql(TARGET_TABLE_DETAILS, conn, if_exists="replace", index=False)
    print(f"Tabela '{TARGET_TABLE_DETAILS}' salva com sucesso no banco.")

    conn.close()
    print("\n Fim do clustering_stats_3_4_5_kmeans7_leagues.py")


if __name__ == "__main__":
    main()
