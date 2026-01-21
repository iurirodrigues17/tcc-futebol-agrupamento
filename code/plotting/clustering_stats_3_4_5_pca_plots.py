import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DB_PATH = "../data/database.sqlite"
SOURCE_TABLE = "team_windows_stats_3_4_5_kmeans3"
PCA_SCORES_TABLE = "clustering_stats_3_4_5_pca_scores"

FIG_DIR = "../figures"


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def load_source_data(conn):
    print("Lendo tabela de origem para PCA:", SOURCE_TABLE)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)
    print("Shape da tabela de origem:", df.shape)

    # Seleciona apenas colunas mean_ / std_ 
    feature_cols = [c for c in df.columns if c.startswith("mean_") or c.startswith("std_")]
    feature_cols = sorted(feature_cols)
    print("\nTotal de colunas mean_/std_ para PCA:", len(feature_cols))
    print("Exemplos de features:", feature_cols[:15])

    X = df[feature_cols].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Shape de X_scaled:", X_scaled.shape)

    return df, X_scaled, feature_cols


def compute_pca(X_scaled, n_components=10):
    print(f"\nRodando PCA com n_components={n_components}...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    print("\n=== Variância explicada (PCA recalculado) ===")
    cum = 0.0
    for i, v in enumerate(var_ratio, start=1):
        cum += v
        print(f"PC{i:02d}: var={v:.4f} (acumulada={cum:.4f})")

    return pca, X_pca, var_ratio


def plot_scree(var_ratio):
    ensure_fig_dir()
    pcs = np.arange(1, len(var_ratio) + 1)

    plt.figure()
    plt.bar(pcs, var_ratio)
    plt.plot(pcs, np.cumsum(var_ratio), marker="o")
    plt.xlabel("Componente Principal")
    plt.ylabel("Variância explicada / acumulada")
    plt.title("PCA – Variância explicada (scree plot)")
    plt.xticks(pcs)
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = os.path.join(FIG_DIR, "pca_scree_stats_3_4_5.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Scree plot salvo em: {out_path}")


def load_pca_scores(conn):
    print("\nLendo tabela de scores PCA:", PCA_SCORES_TABLE)
    df_scores = pd.read_sql_query(f"SELECT * FROM {PCA_SCORES_TABLE};", conn)
    print("Shape de df_scores:", df_scores.shape)
    print("Colunas disponíveis em df_scores:", df_scores.columns.tolist())
    return df_scores


def plot_pc1_pc2_by_result(df_scores):
    ensure_fig_dir()

    if not {"PC1", "PC2", "result_current"}.issubset(df_scores.columns):
        print("⚠ Não encontrei colunas PC1, PC2 e result_current em df_scores. Pulando esse gráfico.")
        return

    # Mapeia rótulos para texto
    label_map = {
        -1: "Derrota",
         0: "Empate",
         1: "Vitória",
    }

    plt.figure()
    for val, name in label_map.items():
        mask = df_scores["result_current"] == val
        if mask.sum() == 0:
            continue
        plt.scatter(
            df_scores.loc[mask, "PC1"],
            df_scores.loc[mask, "PC2"],
            label=name,
            alpha=0.5,
            s=10,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA – PC1 vs PC2 por resultado (Vitória/Empate/Derrota)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = os.path.join(FIG_DIR, "pca_pc1_pc2_by_result.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Gráfico PC1 vs PC2 por resultado salvo em: {out_path}")


def plot_pc1_pc2_by_cluster(df_scores):
    ensure_fig_dir()

    if not {"PC1", "PC2", "cluster_kmeans_3_stats"}.issubset(df_scores.columns):
        print("⚠ Não encontrei colunas PC1, PC2 e cluster_kmeans_3_stats em df_scores. Pulando esse gráfico.")
        return

    plt.figure()
    clusters = sorted(df_scores["cluster_kmeans_3_stats"].unique())

    for c in clusters:
        mask = df_scores["cluster_kmeans_3_stats"] == c
        plt.scatter(
            df_scores.loc[mask, "PC1"],
            df_scores.loc[mask, "PC2"],
            label=f"Cluster {c}",
            alpha=0.5,
            s=10,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA – PC1 vs PC2 por cluster (K-Means k=3)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = os.path.join(FIG_DIR, "pca_pc1_pc2_by_cluster.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Gráfico PC1 vs PC2 por cluster salvo em: {out_path}")


def plot_pc1_pc2_by_league(df_scores):
    """Opcional: PC1 x PC2 colorido por liga."""
    ensure_fig_dir()

    if not {"PC1", "PC2", "league_id"}.issubset(df_scores.columns):
        print("⚠ Não encontrei colunas PC1, PC2 e league_id em df_scores. Pulando esse gráfico.")
        return

    plt.figure()
    leagues = sorted(df_scores["league_id"].unique())

    for lg in leagues:
        mask = df_scores["league_id"] == lg
        plt.scatter(
            df_scores.loc[mask, "PC1"],
            df_scores.loc[mask, "PC2"],
            label=f"Liga {lg}",
            alpha=0.4,
            s=8,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA – PC1 vs PC2 por liga")
    plt.legend(markerscale=2, fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = os.path.join(FIG_DIR, "pca_pc1_pc2_by_league.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Gráfico PC1 vs PC2 por liga salvo em: {out_path}")


def main():
    print("DB_PATH:", DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    # 1) Recalcula PCA só para gerar o scree plot
    _, X_scaled, _ = load_source_data(conn)
    _, _, var_ratio = compute_pca(X_scaled, n_components=10)
    plot_scree(var_ratio)

    # 2) Usa os scores já salvos para PC1/PC2
    df_scores = load_pca_scores(conn)
    conn.close()

    plot_pc1_pc2_by_result(df_scores)
    plot_pc1_pc2_by_cluster(df_scores)
    plot_pc1_pc2_by_league(df_scores)  # opcional

    print("\n Todos os gráficos de PCA foram gerados.")


if __name__ == "__main__":
    main()
