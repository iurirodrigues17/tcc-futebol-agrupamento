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

# CONFIGURAÇÃO DOS EXPERIMENTOS

DB_PATH = "../data/database.sqlite"
SOURCE_TABLE = "team_windows_stats_3_4_5"
TARGET_TABLE = "team_windows_stats_3_4_5_kmeans3"
METRICS_TABLE = "clustering_stats_3_4_5_metrics"

# Flags para ligar/desligar grupos de atributos disciplinares
# (ajuste aqui para cada experimento que deseje rodar)
USE_FOULS = True      # True = mantém mean/std de faltas, False = remove
USE_YELLOWS = True   # True = mantém mean/std de amarelos, False = remove
USE_REDS = True      # True = mantém mean/std de vermelhos, False = remove

# Ligas a serem excluídas
EXCLUDED_LEAGUES = [24558]

def main():
    # 1) Ler tabela de origem
    print("DB_PATH:", DB_PATH)
    print("Tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Tabela carregada:", df.shape)
    print("Colunas de exemplo:", df.columns[:20].tolist())
    
    # Filtro de ligas
    if "league_id" not in df.columns:
        raise ValueError("Coluna 'league_id' não encontrada na tabela de origem. Não é possível filtrar ligas.")

    print("\n=== Filtro de ligas ===")
    print("Ligas excluídas:", EXCLUDED_LEAGUES)
    shape_before = df.shape

    df = df[~df["league_id"].isin(EXCLUDED_LEAGUES)].copy()

    shape_after = df.shape
    removed = shape_before[0] - shape_after[0]
    print(f"Registros removidos pelo filtro de ligas: {removed}")
    print("Shape antes do filtro:", shape_before)
    print("Shape após o filtro:", shape_after)

    # 2) Selecionar apenas colunas mean_ / std_ como base do vetor
    feature_cols = [c for c in df.columns if c.startswith("mean_") or c.startswith("std_")]
    feature_cols = sorted(feature_cols)

    print("\nTotal de colunas mean_/std_ encontradas:", len(feature_cols))

    # Identificar grupos de atributos disciplinares
    fouls_cols = [c for c in feature_cols if "fouls_for" in c]
    yellows_cols = [c for c in feature_cols if "yellows_for" in c]
    reds_cols = [c for c in feature_cols if "reds_for" in c]

    to_drop = []

    # Aplicar flags de uso
    if not USE_FOULS:
        to_drop.extend(fouls_cols)
    if not USE_YELLOWS:
        to_drop.extend(yellows_cols)
    if not USE_REDS:
        to_drop.extend(reds_cols)

    # Remover duplicatas (caso algum nome tenha caído em dois filtros por engano)
    to_drop = sorted(list(set(to_drop)))

    features_selected = [c for c in feature_cols if c not in to_drop]

    print("\nResumo da seleção de features:")
    print(f"  Total original (mean_/std_): {len(feature_cols)}")
    print(f"  Usando faltas?      {USE_FOULS}  -> removidas: {len(fouls_cols) if not USE_FOULS else 0}")
    print(f"  Usando amarelos?    {USE_YELLOWS} -> removidas: {len(yellows_cols) if not USE_YELLOWS else 0}")
    print(f"  Usando vermelhos?   {USE_REDS}    -> removidas: {len(reds_cols) if not USE_REDS else 0}")
    print(f"  Nº de features selecionadas: {len(features_selected)}")
    print("Exemplos de features selecionadas:", features_selected[:15])

    # 3) Construir matriz X e padronizar
    X = df[features_selected].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Shape de X_scaled:", X_scaled.shape)

    # 4) Métricas usando o rótulo REAL (result_current)
    y_true = df["result_current"].values

    print("\n=== Separação usando o rótulo REAL (result_current) ===")
    print("Distribuição de rótulos (result_current):")
    print(df["result_current"].value_counts())

    try:
        sil_real = silhouette_score(X_scaled, y_true)
        ch_real = calinski_harabasz_score(X_scaled, y_true)
        db_real = davies_bouldin_score(X_scaled, y_true)
    except Exception as e:
        print("Erro ao calcular métricas com rótulo real:", e)
        sil_real = np.nan
        ch_real = np.nan
        db_real = np.nan

    print(f"Silhouette (rótulo real): {sil_real:.4f}")
    print(f"Calinski-Harabasz (rótulo real): {ch_real:.4f}")
    print(f"Davies-Bouldin (rótulo real): {db_real:.4f}")

    # 5) KMeans com k=3 (sem usar rótulo no treino)
    print("\n=== KMeans com k=3 (sem usar rótulo no treino) ===")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df["cluster_kmeans_3_stats"] = clusters

    sil_kmeans = silhouette_score(X_scaled, clusters)
    ch_kmeans = calinski_harabasz_score(X_scaled, clusters)
    db_kmeans = davies_bouldin_score(X_scaled, clusters)

    ari = adjusted_rand_score(y_true, clusters)
    nmi = normalized_mutual_info_score(y_true, clusters)

    print(f"Silhouette (KMeans): {sil_kmeans:.4f}")
    print(f"Calinski-Harabasz (KMeans): {ch_kmeans:.4f}")
    print(f"Davies-Bouldin (KMeans): {db_kmeans:.4f}")
    print(f"Adjusted Rand Index (vs rótulo real): {ari:.4f}")
    print(f"NMI (vs rótulo real): {nmi:.4f}")

    # Distribuição de clusters
    print("\nDistribuição de clusters (KMeans k=3):")
    print(df["cluster_kmeans_3_stats"].value_counts())

    # 6) Crosstab clusters x resultado atual
    print("\nCrosstab clusters x resultado atual (contagem absoluta):")
    ct_abs = pd.crosstab(df["cluster_kmeans_3_stats"], df["result_current"])
    print(ct_abs)

    print("\nCrosstab clusters x resultado atual (proporção por cluster):")
    ct_prop = pd.crosstab(df["cluster_kmeans_3_stats"], df["result_current"], normalize="index")
    print(ct_prop)

    # 7) Salvar tabela de saída com clusters
    df.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
    print(f"\nTabela com clusters salva como: {TARGET_TABLE}")

    # 8) Salvar tabela de métricas
    metrics_row = {
        "n_samples": len(df),
        "n_features": len(features_selected),
        "use_fouls": USE_FOULS,
        "use_yellows": USE_YELLOWS,
        "use_reds": USE_REDS,
        "sil_real": sil_real,
        "ch_real": ch_real,
        "db_real": db_real,
        "sil_kmeans": sil_kmeans,
        "ch_kmeans": ch_kmeans,
        "db_kmeans": db_kmeans,
        "ari": ari,
        "nmi": nmi,
    }

    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_sql(METRICS_TABLE, conn, if_exists="replace", index=False)
    print(f"Tabela de métricas salva como: {METRICS_TABLE}")

    conn.close()
    print("Fim do clustering_stats_3_4_5.py")


if __name__ == "__main__":
    main()
