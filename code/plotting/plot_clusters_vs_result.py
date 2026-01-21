import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "../data/database.sqlite"

# pasta onde as figuras vão ser salvas
OUTPUT_DIR = "../figures"

# Experimentos que fazem sentido para crosstab cluster x resultado
EXPERIMENTS = [
    ("team_windows_hist5_kmeans3", "Histórico de 5 partidas (hist5)"),
    ("team_windows_hist5_reduced_kmeans3", "Histórico de 5 partidas (vetor reduzido)"),
    ("team_windows_stats_3_4_5_kmeans3", "3, 4 e 5 partidas (médias e desvios)")
]

def get_cluster_column(columns):
    """
    Tenta descobrir automaticamente o nome da coluna de cluster.
    """
    if "cluster_kmeans_3_stats" in columns:
        return "cluster_kmeans_3_stats"
    if "cluster_kmeans_3" in columns:
        return "cluster_kmeans_3"
    if "cluster" in columns:
        return "cluster"
    return None

def main():
    # garante que a pasta de saída exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    for table_name, label in EXPERIMENTS:
        print(f"=== Plotando experimento: {table_name} ===")

        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            print(f"  Erro ao ler {table_name}: {e}")
            continue

        cluster_col = get_cluster_column(df.columns)

        missing = []
        if "result_current" not in df.columns:
            missing.append("result_current")
        if cluster_col is None:
            missing.append("coluna de cluster")

        if missing:
            print(f"  Pulando {table_name} – colunas ausentes: {missing}")
            continue

        # Crosstab normalizada por cluster (cada linha = 1)
        ctab = pd.crosstab(df[cluster_col], df["result_current"], normalize="index")

        plt.figure(figsize=(7, 5))
        sns.heatmap(ctab, annot=True, fmt=".2f", cmap="Blues")

        plt.title(
            f"{label}: distribuiçãopor cluster (k=3)",
            pad=20
        )
        plt.xlabel("Resultado da partida (rótulo real)")
        plt.ylabel("Cluster")

        # Ajuste para não cortar o título
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{table_name}_cluster_vs_result.png"
        )
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"  Figura salva em: {output_path}\n")

    conn.close()

if __name__ == "__main__":
    main()
