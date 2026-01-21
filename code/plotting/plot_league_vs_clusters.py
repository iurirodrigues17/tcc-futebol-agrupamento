# arquivo: plotting/plot_league_vs_clusters.py

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = "../data/database.sqlite"
TABLE = "team_windows_stats_3_4_5_clusters_vs_league" # "team_windows_stats_3_4_5_kmeans3"   ou 'team_windows_stats_3_4_5_clusters_vs_league'
CLUSTER_COL = "cluster_kmeans_3_stats"
LEAGUE_COL = "league_id"
OUTPUT_DIR = "../figures"
FIG_NAME = "league_vs_clusters_stats_3_4_5.png"
TITLE = "Clusters por liga – vetor de 3, 4 e 5 partidas"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT {LEAGUE_COL}, {CLUSTER_COL} FROM {TABLE}", conn)
    conn.close()

    league_map = {
    1729: "England Premier League",
    4769: "France Ligue 1",
    7809: "Germany 1. Bundesliga",
    10257: "Italy Serie A",
    13274: "Netherlands Eredivisie",
    21518: "Spain LIGA BBVA",
}
    df["league_name"] = df[LEAGUE_COL].map(league_map)
    # e usar 'league_name' no lugar de LEAGUE_COL na crosstab


    # Crosstab liga x cluster (contagem)
    ct_counts = pd.crosstab(df[LEAGUE_COL], df[CLUSTER_COL])

    # Proporção por liga (linha normalizada)
    ct_props = ct_counts.div(ct_counts.sum(axis=1), axis=0)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        ct_props,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True
    )
    plt.title(TITLE)
    plt.xlabel("Cluster K-Means (k=3)")
    plt.ylabel("Liga (league_id)")
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, FIG_NAME)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figura salva em: {fig_path}")


if __name__ == "__main__":
    main()
