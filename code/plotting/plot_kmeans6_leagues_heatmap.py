import os
import sqlite3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para o banco
DB_PATH = "../data/database.sqlite"

# Pasta de saída para as figuras
OUTPUT_DIR = "../figures"

# Mapeamento id
LEAGUE_NAMES = {
    1729: "ENG PL",   # England Premier League
    4769: "FRA L1",   # France Ligue 1
    7809: "GER BL",   # Germany 1. Bundesliga
    10257: "ITA SA",  # Italy Serie A
    13274: "NED ERE", # Netherlands Eredivisie
    21518: "ESP LL",  # Spain La Liga
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    # Tabela gerada pelo clustering_stats_3_4_5_kmeans6_leagues.py
    query = """
        SELECT league_id, cluster_kmeans_6_leagues AS cluster
        FROM team_windows_stats_3_4_5_kmeans6_leagues
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("DataFrame vazio, verifique o nome da tabela ou o caminho do banco.")
        return

    # Mantém só as ligas de interesse
    df = df[df["league_id"].isin(LEAGUE_NAMES.keys())].copy()

    # Tabela liga x cluster (contagem absoluta)
    crosstab_abs = pd.crosstab(df["league_id"], df["cluster"])

    # Proporção por liga (linha normalizada)
    crosstab_prop = crosstab_abs.div(crosstab_abs.sum(axis=1), axis=0) * 100

    # Renomeia índices com nomes das ligas
    crosstab_prop.index = crosstab_prop.index.map(LEAGUE_NAMES)

    # Garante ordem das colunas de cluster: 0..5
    crosstab_prop = crosstab_prop.reindex(sorted(crosstab_prop.columns), axis=1)

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        crosstab_prop,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "% de janelas da liga no cluster"},
    )

    plt.xlabel("Cluster (K-Means, k = 6)")
    plt.ylabel("Liga")
    plt.title("Distribuição das janelas de desempenho por liga e cluster (k = 6)")

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "league_vs_clusters_stats_3_4_5_kmeans6.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figura salva em: {output_path}")

if __name__ == "__main__":
    main()
