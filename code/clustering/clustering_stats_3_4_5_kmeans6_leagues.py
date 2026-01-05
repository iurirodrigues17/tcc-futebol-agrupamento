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
TABLE_SOURCE = "team_windows_stats_3_4_5"

# k=6
K_LEAGUES = 6

# Liga a ser excluída
EXCLUDED_LEAGUES = [24558]

# Flags para ligar/desligar grupos de atributos disciplinares
# (ajuste aqui para cada experimento que deseje rodar)
USE_FOULS = True      # True = mantém mean/std de faltas, False = remove
USE_YELLOWS = True   # True = mantém mean/std de amarelos, False = remove
USE_REDS = True      # True = mantém mean/std de vermelhos, False = remove


def main():
    print(f"DB_PATH: {DB_PATH}")
    print(f"Lendo tabela de origem: {TABLE_SOURCE}")

    conn = sqlite3.connect(DB_PATH)
    
    # 1) Leitura da tabela
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_SOURCE}", conn)
    print(f"Shape da tabela de origem: {df.shape}")
    print("Colunas de exemplo:", list(df.columns[:20]))

    # 2) Filtro de ligas
    print("\n=== Filtro de ligas ===")
    shape_before = df.shape
    df = df[~df["league_id"].isin(EXCLUDED_LEAGUES)].copy()
    shape_after = df.shape
    removed = shape_before[0] - shape_after[0]
    print(f"Ligas excluídas: {EXCLUDED_LEAGUES}")
    print(f"Registros removidos pelo filtro de ligas: {removed}")
    print(f"Shape antes do filtro: {shape_before}")
    print(f"Shape após o filtro: {shape_after}")

    # 3) Seleção de features mean_/std_
    all_cols = list(df.columns)

    # Todas as colunas que começam com mean_ ou std_
    mean_std_cols = [c for c in all_cols if c.startswith("mean_") or c.startswith("std_")]
    total_mean_std = len(mean_std_cols)
    print(f"\nTotal de colunas mean_/std_ encontradas: {total_mean_std}")

    # Filtrar de acordo com uso de faltas / amarelos / vermelhos
    selected_cols = mean_std_cols.copy()

    # Faltas
    if not USE_FOULS:
        removed_fouls = [c for c in selected_cols if "fouls" in c]
        selected_cols = [c for c in selected_cols if "fouls" not in c]
    else:
        removed_fouls = []

    # Cartões amarelos
    if not USE_YELLOWS:
        removed_yellows = [c for c in selected_cols if "yellows" in c]
        selected_cols = [c for c in selected_cols if "yellows" not in c]
    else:
        removed_yellows = []

    # Cartões vermelhos
    if not USE_REDS:
        removed_reds = [c for c in selected_cols if "reds" in c]
        selected_cols = [c for c in selected_cols if "reds" not in c]
    else:
        removed_reds = []

    print("\nResumo da seleção de features:")
    print(f"  Total original (mean_/std_): {total_mean_std}")
    print(f"  Usando faltas?      {USE_FOULS}  -> removidas: {len(removed_fouls)}")
    print(f"  Usando amarelos?    {USE_YELLOWS} -> removidas: {len(removed_yellows)}")
    print(f"  Usando vermelhos?   {USE_REDS}    -> removidas: {len(removed_reds)}")
    print(f"  Nº de features selecionadas: {len(selected_cols)}")
    print("  Exemplos de features selecionadas:", selected_cols[:15])

    # 4) Padronização
    X = df[selected_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Shape de X_scaled: {X_scaled.shape}")

    # 5) KMeans com k = 6
    print(f"\n=== KMeans com k={K_LEAGUES} (agrupando janelas em 'tipos de liga') ===")

    kmeans = KMeans(
        n_clusters=K_LEAGUES,
        random_state=42,
        n_init=10,
    )
    clusters = kmeans.fit_predict(X_scaled)

    df["cluster_kmeans_6_leagues"] = clusters

    sil = silhouette_score(X_scaled, clusters)
    ch = calinski_harabasz_score(X_scaled, clusters)
    db = davies_bouldin_score(X_scaled, clusters)

    print(f"Silhouette (KMeans k={K_LEAGUES}): {sil:.4f}")
    print(f"Calinski-Harabasz (KMeans k={K_LEAGUES}): {ch:.4f}")
    print(f"Davies-Bouldin (KMeans k={K_LEAGUES}): {db:.4f}")

    # 6) Métricas clusters vs ligas
    print("\n=== Métricas clusters vs ligas (usando k=6) ===")

    ari = adjusted_rand_score(df["league_id"], df["cluster_kmeans_6_leagues"])
    nmi = normalized_mutual_info_score(df["league_id"], df["cluster_kmeans_6_leagues"])

    print(f"ARI (clusters vs league_id): {ari:.4f}")
    print(f"NMI (clusters vs league_id): {nmi:.4f}")

    league_counts = df["league_id"].value_counts().sort_index()
    print("\nDistribuição geral de ligas (contagem):")
    print(league_counts)

    # crosstab cluster x liga
    ct = pd.crosstab(df["cluster_kmeans_6_leagues"], df["league_id"])
    print("\nCrosstab cluster x liga (contagem absoluta):")
    print(ct)

    # proporção por cluster
    ct_cluster_prop = ct.div(ct.sum(axis=1), axis=0)
    print("\nCrosstab cluster x liga (proporção por cluster):")
    print(ct_cluster_prop)

    # proporção por liga
    ct_league_prop = ct.div(ct.sum(axis=0), axis=1)
    print("\nCrosstab liga x cluster (proporção por liga):")
    print(ct_league_prop.T)

    # 7) Salvar resultados no banco
    
    # Tabela com as janelas + cluster de liga
    target_table_clusters = "team_windows_stats_3_4_5_kmeans6_leagues"
    df.to_sql(target_table_clusters, conn, if_exists="replace", index=False)
    print(f"\nTabela '{target_table_clusters}' salva com sucesso no banco.")

    # Tabela de métricas gerais
    metrics_df = pd.DataFrame(
        {
            "k": [K_LEAGUES],
            "silhouette": [sil],
            "calinski_harabasz": [ch],
            "davies_bouldin": [db],
            "ari_vs_league": [ari],
            "nmi_vs_league": [nmi],
            "n_registros": [len(df)],
        }
    )

    target_table_metrics = "clustering_stats_3_4_5_kmeans6_leagues_metrics"
    metrics_df.to_sql(target_table_metrics, conn, if_exists="replace", index=False)
    print(f"Tabela '{target_table_metrics}' salva com sucesso no banco.")

    # Tabela de contagens e proporções liga x cluster
    rows = []
    for cluster_id in ct.index:
        for league_id in ct.columns:
            count = ct.loc[cluster_id, league_id]
            prop_cluster = ct_cluster_prop.loc[cluster_id, league_id]
            prop_league = ct_league_prop.loc[cluster_id, league_id]
            rows.append(
                {
                    "cluster_kmeans_6_leagues": int(cluster_id),
                    "league_id": int(league_id),
                    "count": int(count),
                    "prop_within_cluster": float(prop_cluster),
                    "prop_within_league": float(prop_league),
                }
            )

    ct_long_df = pd.DataFrame(rows)
    target_table_ct = "team_windows_stats_3_4_5_kmeans6_leagues_vs_league"
    ct_long_df.to_sql(target_table_ct, conn, if_exists="replace", index=False)
    print(f"Tabela '{target_table_ct}' salva com sucesso no banco.")

    conn.close()
    print("\n Fim do clustering_stats_3_4_5_kmeans6_leagues.py")


if __name__ == "__main__":
    main()
