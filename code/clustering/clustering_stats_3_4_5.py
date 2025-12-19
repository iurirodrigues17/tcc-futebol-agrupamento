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
TARGET_TABLE = "team_windows_stats_3_4_5_kmeans3"

USE_FOULS = False      # mean_fouls_for_*, std_fouls_for_*
USE_YELLOWS = True    # mean_yellows_for_*, std_yellows_for_*
USE_REDS = True       # mean_reds_for_*, std_reds_for_*


def main():
    print("DB_PATH:", DB_PATH)
    print("Tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Tabela carregada:", df.shape)
    print("Colunas de exemplo:", df.columns[:20].tolist())

    # 1) Colunas de CONTEXTO (não entram no X)
    context_cols = [
        "team_id",
        "current_match_id",
        "current_date",
        "league_id",
        "season",
        "opponent_id_current",
        "result_current",
    ]
    context_cols = [c for c in context_cols if c in df.columns]

    if "result_current" not in df.columns:
        raise ValueError("Coluna 'result_current' não encontrada na tabela.")

    # 2) FEATURES: todas mean_/std_ e depois filtramos faltas/cartões via flags
    feature_cols_all = [
        c for c in df.columns
        if (c.startswith("mean_") or c.startswith("std_")) and (c not in context_cols)
    ]

    if len(feature_cols_all) == 0:
        raise ValueError(
            "Nenhuma feature encontrada com prefixo mean_ ou std_. "
            "Confira se a tabela possui colunas como mean_goals_for_last5, std_shots_for_last3, etc."
        )

    # filtro pelas flags
    feature_cols = []
    dropped_fouls = []
    dropped_yellows = []
    dropped_reds = []

    for c in feature_cols_all:
        c_low = c.lower()

        # faltas
        if ("fouls_for" in c_low) and (not USE_FOULS):
            dropped_fouls.append(c)
            continue

        # cartões amarelos
        if ("yellows_for" in c_low) and (not USE_YELLOWS):
            dropped_yellows.append(c)
            continue

        # cartões vermelhos
        if ("reds_for" in c_low) and (not USE_REDS):
            dropped_reds.append(c)
            continue

        feature_cols.append(c)

    print("\nResumo da seleção de features:")
    print(f"  Total original (mean_/std_): {len(feature_cols_all)}")
    print(f"  Usando faltas?      {USE_FOULS}  -> removidas: {len(dropped_fouls)}")
    print(f"  Usando amarelos?    {USE_YELLOWS} -> removidas: {len(dropped_yellows)}")
    print(f"  Usando vermelhos?   {USE_REDS}    -> removidas: {len(dropped_reds)}")
    print(f"  Nº de features selecionadas: {len(feature_cols)}")
    print("Exemplos de features selecionadas:", feature_cols[:15])

    # 3) Preparar X (tratando NaN) e padronizar (z-score)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Shape de X_scaled:", X_scaled.shape)

    # Rótulo real: -1, 0, 1 (derrota, empate, vitória)
    y_true = df["result_current"].values

    # 4) Métricas com o RÓTULO REAL
    print("\n=== Separação usando o rótulo REAL (result_current) ===")
    print("Distribuição de rótulos (result_current):")
    print(pd.Series(y_true).value_counts())

    try:
        sil_real = silhouette_score(X_scaled, y_true)
        ch_real = calinski_harabasz_score(X_scaled, y_true)
        db_real = davies_bouldin_score(X_scaled, y_true)
    except Exception as e:
        print("Erro ao calcular métricas com rótulo real:", e)
        sil_real = ch_real = db_real = np.nan

    print(f"Silhouette (rótulo real): {sil_real:.4f}")
    print(f"Calinski-Harabasz (rótulo real): {ch_real:.4f}")
    print(f"Davies-Bouldin (rótulo real): {db_real:.4f}")

    # 5) KMeans k=3 (sem usar rótulo no treino)
    print("\n=== KMeans com k=3 (sem usar rótulo no treino) ===")
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
    print(f"Adjusted Rand Index (vs rótulo real): {ari:.4f}")
    print(f"NMI (vs rótulo real): {nmi:.4f}")
    print("\nDistribuição de clusters (KMeans k=3):")
    print(pd.Series(labels_km, name="cluster_kmeans_3_stats").value_counts())

    # 6) Crosstab cluster x resultado da partida
    df["cluster_kmeans_3_stats"] = labels_km

    print("\nCrosstab clusters x resultado atual (contagem absoluta):")
    print(pd.crosstab(df["cluster_kmeans_3_stats"], df["result_current"]))

    print("\nCrosstab clusters x resultado atual (proporção por cluster):")
    ct = pd.crosstab(df["cluster_kmeans_3_stats"], df["result_current"], normalize="index")
    print(ct)

    # 7) Salvar tabela de saída
    df.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
    print(f"\nTabela com clusters salva como: {TARGET_TABLE}")

    conn.close()
    print("Fim do clustering_stats_3_4_5.py")


if __name__ == "__main__":
    main()
