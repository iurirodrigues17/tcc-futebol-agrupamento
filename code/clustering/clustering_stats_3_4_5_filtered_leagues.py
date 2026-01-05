# clustering_stats_3_4_5.py
# Agrupamento global (vitória/empate/derrota) usando apenas estatísticas mean/std
# + filtro para remover ligas com menos jogos (Eredivisie 13274 e Primeira Liga 24558)

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
TARGET_TABLE = "team_windows_stats_3_4_5_kmeans3"
METRICS_TABLE = "clustering_stats_3_4_5_metrics"

# ----------------------------------------------------------------------
# Configurações de features
# ----------------------------------------------------------------------

# Melhor setup que vimos até agora para o clustering global:
#   - usar faltas
#   - NÃO usar cartões amarelos
#   - NÃO usar cartões vermelhos
USE_FOULS = True
USE_YELLOWS = False
USE_REDS = False

# Ligas a excluir (menos jogos / menor número de janelas)
EXCLUDED_LEAGUES = [13274, 24558]


def main():
    print("DB_PATH:", DB_PATH)
    print("Tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Tabela carregada:", df.shape)
    print("Colunas de exemplo:", df.columns[:20].tolist())

    # ------------------------------------------------------------------
    # Filtro de ligas (remover Eredivisie 13274 e Primeira Liga 24558)
    # ------------------------------------------------------------------
    if "league_id" not in df.columns:
        raise ValueError("Coluna 'league_id' não encontrada na tabela de origem.")

    before_shape = df.shape
    mask_excluded = df["league_id"].isin(EXCLUDED_LEAGUES)
    removed_count = mask_excluded.sum()
    df = df[~mask_excluded].reset_index(drop=True)
    after_shape = df.shape

    print("\n=== Filtro de ligas ===")
    print("Ligas excluídas:", EXCLUDED_LEAGUES)
    print(f"Registros removidos: {removed_count}")
    print("Shape antes do filtro:", before_shape)
    print("Shape após o filtro:", after_shape)

    # ------------------------------------------------------------------
    # Separar rótulo real
    # ------------------------------------------------------------------
    if "result_current" not in df.columns:
        raise ValueError("Coluna 'result_current' não encontrada na tabela!")

    y_real = df["result_current"].values  # -1, 0, 1

    # ------------------------------------------------------------------
    # Seleção de features: apenas colunas mean_ / std_
    # ------------------------------------------------------------------
    all_mean_std_cols = [
        c for c in df.columns if c.startswith("mean_") or c.startswith("std_")
    ]
    total_original = len(all_mean_std_cols)

    # Prefixos dos atributos que podem ser ligados/desligados
    foul_prefixes = ["mean_fouls_for_last", "std_fouls_for_last"]
    yellow_prefixes = ["mean_yellows_for_last", "std_yellows_for_last"]
    red_prefixes = ["mean_reds_for_last", "std_reds_for_last"]

    def remove_prefix_group(cols, prefixes):
        to_remove = []
        for p in prefixes:
            for c in cols:
                if c.startswith(p):
                    to_remove.append(c)
        return [c for c in cols if c not in to_remove], len(to_remove)

    feature_cols = list(all_mean_std_cols)

    removed_fouls = 0
    removed_yellows = 0
    removed_reds = 0

    if not USE_FOULS:
        feature_cols, removed_fouls = remove_prefix_group(feature_cols, foul_prefixes)
    if not USE_YELLOWS:
        feature_cols, removed_yellows = remove_prefix_group(feature_cols, yellow_prefixes)
    if not USE_REDS:
        feature_cols, removed_reds = remove_prefix_group(feature_cols, red_prefixes)

    print("\nResumo da seleção de features:")
    print(f"  Total original (mean_/std_): {total_original}")
    print(f"  Usando faltas?      {USE_FOULS}  -> removidas: {removed_fouls}")
    print(f"  Usando amarelos?    {USE_YELLOWS} -> removidas: {removed_yellows}")
    print(f"  Usando vermelhos?   {USE_REDS}    -> removidas: {removed_reds}")
    print(f"  Nº de features selecionadas: {len(feature_cols)}")

    if len(feature_cols) == 0:
        raise ValueError("Nenhuma feature selecionada! Verifique as flags de uso de atributos.")

    print("Exemplos de features selecionadas:", feature_cols[:15])

    # ------------------------------------------------------------------
    # Montar matriz X e padronizar
    # ------------------------------------------------------------------
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Shape de X_scaled:", X_scaled.shape)

    # ------------------------------------------------------------------
    # Métricas usando o rótulo REAL como 'cluster'
    # ------------------------------------------------------------------
    print("\n=== Separação usando o rótulo REAL (result_current) ===")
    unique, counts = np.unique(y_real, return_counts=True)
    distrib_real = pd.Series(counts, index=unique, name="count")
    print("Distribuição de rótulos (result_current):")
    print(distrib_real)

    # Para métricas de cluster, precisamos de pelo menos 2 classes distintas
    if len(unique) >= 2:
        try:
            sil_real = silhouette_score(X_scaled, y_real)
        except Exception:
            sil_real = np.nan
        try:
            ch_real = calinski_harabasz_score(X_scaled, y_real)
        except Exception:
            ch_real = np.nan
        try:
            db_real = davies_bouldin_score(X_scaled, y_real)
        except Exception:
            db_real = np.nan
    else:
        sil_real = ch_real = db_real = np.nan

    print(f"Silhouette (rótulo real): {sil_real:0.4f}")
    print(f"Calinski-Harabasz (rótulo real): {ch_real:0.4f}")
    print(f"Davies-Bouldin (rótulo real): {db_real:0.4f}")

    # ------------------------------------------------------------------
    # KMeans com k=3 (vitória, empate, derrota - sem usar o rótulo no treino)
    # ------------------------------------------------------------------
    print("\n=== KMeans com k=3 (sem usar rótulo no treino) ===")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Métricas de cluster
    sil_k = silhouette_score(X_scaled, cluster_labels)
    ch_k = calinski_harabasz_score(X_scaled, cluster_labels)
    db_k = davies_bouldin_score(X_scaled, cluster_labels)

    # Comparação com rótulo real
    ari = adjusted_rand_score(y_real, cluster_labels)
    nmi = normalized_mutual_info_score(y_real, cluster_labels)

    print(f"Silhouette (KMeans): {sil_k:0.4f}")
    print(f"Calinski-Harabasz (KMeans): {ch_k:0.4f}")
    print(f"Davies-Bouldin (KMeans): {db_k:0.4f}")
    print(f"Adjusted Rand Index (vs rótulo real): {ari:0.4f}")
    print(f"NMI (vs rótulo real): {nmi:0.4f}")

    # Distribuição dos clusters
    s_clusters = pd.Series(cluster_labels, name="cluster_kmeans_3_stats").value_counts().sort_index()
    print("\nDistribuição de clusters (KMeans k=3):")
    print(s_clusters)

    # Crosstab cluster x resultado atual
    df_out = df.copy()
    df_out["cluster_kmeans_3_stats"] = cluster_labels

    crosstab_abs = pd.crosstab(df_out["cluster_kmeans_3_stats"], df_out["result_current"])
    print("\nCrosstab clusters x resultado atual (contagem absoluta):")
    print(crosstab_abs)

    crosstab_prop = crosstab_abs.div(crosstab_abs.sum(axis=1), axis=0)
    print("\nCrosstab clusters x resultado atual (proporção por cluster):")
    print(crosstab_prop)

    # ------------------------------------------------------------------
    # Salvar tabela com clusters e tabela de métricas
    # ------------------------------------------------------------------
    df_out.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
    print(f"\nTabela com clusters salva como: {TARGET_TABLE}")

    metrics_df = pd.DataFrame(
        {
            "n_samples": [X_scaled.shape[0]],
            "n_features": [X_scaled.shape[1]],
            "use_fouls": [USE_FOULS],
            "use_yellows": [USE_YELLOWS],
            "use_reds": [USE_REDS],
            "excluded_leagues": [",".join(map(str, EXCLUDED_LEAGUES))],
            "sil_real": [sil_real],
            "ch_real": [ch_real],
            "db_real": [db_real],
            "sil_kmeans": [sil_k],
            "ch_kmeans": [ch_k],
            "db_kmeans": [db_k],
            "ari": [ari],
            "nmi": [nmi],
        }
    )

    metrics_df.to_sql(METRICS_TABLE, conn, if_exists="replace", index=False)
    print(f"Tabela de métricas salva como: {METRICS_TABLE}")

    conn.close()
    print("Fim do clustering_stats_3_4_5.py")


if __name__ == "__main__":
    main()
