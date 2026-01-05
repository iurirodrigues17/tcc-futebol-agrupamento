import sqlite3
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

DB_PATH = "../data/database.sqlite"
SOURCE_TABLE = "team_windows_stats_3_4_5_kmeans3"  # tabela com clusters globais
METRICS_TABLE = "clustering_stats_3_4_5_clusters_vs_league_metrics"
DETAIL_TABLE = "team_windows_stats_3_4_5_clusters_vs_league"

def main():
    print("DB_PATH:", DB_PATH)
    print("Lendo tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)
    conn.close()

    print("Tabela carregada:", df.shape)
    print("Colunas disponíveis:", df.columns[:20].tolist())
    
    #Filtro de ligas
    EXCLUDED_LEAGUES = [24558]

    before_filter = len(df)
    df = df[~df["league_id"].isin(EXCLUDED_LEAGUES)].copy()
    after_filter = len(df)

    print("\n=== Filtro de ligas ===")
    print(f"Ligas excluídas: {EXCLUDED_LEAGUES}")
    print(f"Registros removidos pelo filtro de ligas: {before_filter - after_filter}")
    print(f"Shape antes do filtro: ({before_filter}, {df.shape[1]})")
    print(f"Shape após o filtro: ({after_filter}, {df.shape[1]})\n")


    if "league_id" not in df.columns or "cluster_kmeans_3_stats" not in df.columns:
        raise ValueError("A tabela de origem precisa ter 'league_id' e 'cluster_kmeans_3_stats'.")

    # Só para garantir tipos
    df["league_id"] = df["league_id"].astype(str)
    df["cluster_kmeans_3_stats"] = df["cluster_kmeans_3_stats"].astype(int)

    # 1) Métricas: quão “próximo” cluster está de separar por liga
    y_league = df["league_id"]
    y_cluster = df["cluster_kmeans_3_stats"]

    ari_league = adjusted_rand_score(y_league, y_cluster)
    nmi_league = normalized_mutual_info_score(y_league, y_cluster)

    print("\n=== Métricas clusters vs ligas ===")
    print(f"ARI (clusters vs ligas): {ari_league:.4f}")
    print(f"NMI (clusters vs ligas): {nmi_league:.4f}")

    # 2) Distribuição geral de ligas
    print("\nDistribuição geral de ligas (contagem):")
    print(df["league_id"].value_counts())

    # 3) Distribuição de ligas por cluster
    print("\nCrosstab cluster x liga (contagem absoluta):")
    ctab_cluster_league = pd.crosstab(df["cluster_kmeans_3_stats"], df["league_id"])
    print(ctab_cluster_league)

    print("\nCrosstab cluster x liga (proporção por cluster):")
    ctab_cluster_league_prop = ctab_cluster_league.div(ctab_cluster_league.sum(axis=1), axis=0)
    print(ctab_cluster_league_prop)

    # 4) Distribuição de clusters por liga
    print("\nCrosstab liga x cluster (proporção por liga):")
    ctab_league_cluster_prop = pd.crosstab(df["league_id"], df["cluster_kmeans_3_stats"], normalize="index")
    print(ctab_league_cluster_prop)

    # 5) Salvar um resumo no banco
    conn = sqlite3.connect(DB_PATH)

    # Métricas em tabela própria
    metrics_df = pd.DataFrame(
        {
            "ari_clusters_vs_league": [ari_league],
            "nmi_clusters_vs_league": [nmi_league],
            "n_registros": [len(df)],
        }
    )
    metrics_df.to_sql(METRICS_TABLE, conn, if_exists="replace", index=False)

    # Detalhes mínimos
    detail_cols = [
        col for col in df.columns
        if col in ["team_id", "current_match_id", "current_date", "league_id", "season", "cluster_kmeans_3_stats"]
    ]
    df[detail_cols].to_sql(DETAIL_TABLE, conn, if_exists="replace", index=False)

    conn.close()

    print(f"\nResumo de métricas salvo na tabela '{METRICS_TABLE}'.")
    print(f"Detalhes (liga x cluster) salvos na tabela '{DETAIL_TABLE}'.")
    print("Fim do clustering_stats_3_4_5_cluster_vs_league.py")

if __name__ == "__main__":
    main()
