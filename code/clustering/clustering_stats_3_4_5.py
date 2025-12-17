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

def main():
    print("DB_PATH:", DB_PATH)
    print("Tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Tabela carregada:", df.shape)
    print("Colunas de exemplo:", df.columns[:20].tolist())

    # 1) Definir colunas de CONTEXTO
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

    # 2) Definir colunas REDUNDANTES (NÃO entram como features de clustering)
    redundant_cols = [
        "goals_for_last5",
        "shots_for_last5",
        "shots_on_for_last5",
        "yellows_for_last5",
        "reds_for_last5",
        "avg_possession_for_last5",
    ]
    redundant_cols = [c for c in redundant_cols if c in df.columns]
    
    # 3) Features = tudo que não é contexto
    feature_cols = [
        c for c in df.columns
        if c not in context_cols and c not in redundant_cols
    ]

    print(f"Nº de features (sem redundâncias): {len(feature_cols)}")

    # Verificação básica
    if "result_current" not in df.columns:
        raise ValueError("Coluna 'result_current' não encontrada na tabela.")

    # 4) Montar X e padronizar (z-score)
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Shape de X_scaled:", X_scaled.shape)

    # Rótulo real: -1, 0, 1 (derrota, empate, vitória)
    y_true = df["result_current"].values

    # 5) Métricas com o RÓTULO REAL
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

    # 6) KMeans k=3 (sem usar rótulo no treino)
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

    # Distribuição de clusters
    print("\nDistribuição de clusters (KMeans k=3):")
    print(pd.Series(labels_km, name="cluster_kmeans_3_stats").value_counts())

    # Crosstab cluster x resultado da partida
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
