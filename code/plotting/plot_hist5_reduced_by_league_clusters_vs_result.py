import os
import sqlite3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DB_PATH = "../data/database.sqlite"
OUTPUT_DIR = "../figures"

# nomes das ligas
LEAGUE_NAMES = {
    1729: "ENG PL",
    4769: "FRA L1",
    7809: "GER 1.BUN",
    10257: "ITA Serie A",
    13274: "NED Eredivisie",
    21518: "ESP LaLiga",
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    # Tabela criada pelo script clustering_hist5_reduced_by_league.py
    query = """
        SELECT
            league_id,
            result_current,
            cluster_kmeans_3
        FROM team_windows_hist5_reduced_kmeans3_by_league
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # mapeia rótulos -1/0/1 para texto
    result_map = {
        -1: "Derrota",
        0: "Empate",
        1: "Vitória",
    }
    df["resultado_label"] = df["result_current"].map(result_map)

    leagues = sorted(df["league_id"].unique())
    n_leagues = len(leagues)

    ncols = 3
    nrows = (n_leagues + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for ax, league_id in zip(axes.flat, leagues):
        sub = df[df["league_id"] == league_id].copy()

        # matriz de proporções: cluster x resultado (normalizado por linha)
        ctab = pd.crosstab(
            sub["cluster_kmeans_3"],
            sub["resultado_label"],
            normalize="index",
        )

        # garante ordem de colunas
        ctab = ctab.reindex(columns=["Derrota", "Empate", "Vitória"])

        sns.heatmap(
            ctab,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
        )

        ax.set_title(LEAGUE_NAMES.get(league_id, str(league_id)))
        ax.set_xlabel("Resultado atual")
        ax.set_ylabel("Cluster K-Means")

    # desliga eixos sobrando, se houver
    for ax in axes.flat[len(leagues):]:
        ax.axis("off")

    fig.suptitle(
        "Distribuição de resultados por cluster em cada liga\n"
        "(vetor team_windows_hist5_reduced)",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = os.path.join(
        OUTPUT_DIR,
        "hist5_reduced_by_league_clusters_vs_result.png",
    )
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Figura salva em: {output_path}")


if __name__ == "__main__":
    main()
