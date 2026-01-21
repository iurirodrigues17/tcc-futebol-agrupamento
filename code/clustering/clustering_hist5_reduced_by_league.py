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
TABLE = "team_windows_hist5_reduced"
K = 3  # 3 grupos (vitória, empate, derrota)

LEAGUE_NAMES = {
    1729: "England Premier League",
    4769: "France Ligue 1",
    7809: "Germany 1. Bundesliga",
    10257: "Italy Serie A",
    13274: "Netherlands Eredivisie",
    21518: "Spain LIGA BBVA",
    # 24558: "Switzerland Super League",
}

print("DB_PATH:", DB_PATH)
print("Tabela de origem:", TABLE)

EXCLUDED_LEAGUES = [24558]

# 1. Carregar dados completos
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f"SELECT * FROM {TABLE};", conn)
conn.close()

print("Tabela carregada:", df.shape)

# === Filtro de ligas ===
print("\n=== Filtro de ligas ===")
print(f"Ligas excluídas: {EXCLUDED_LEAGUES}")
before_filter = len(df)
df = df[~df["league_id"].isin(EXCLUDED_LEAGUES)].copy()
after_filter = len(df)
print(f"Registros removidos pelo filtro de ligas: {before_filter - after_filter}")
print(f"Shape após o filtro: {df.shape}")

leagues = sorted(df["league_id"].unique())
print("\nLigas disponíveis (após filtro):", leagues)

context_cols = [
    "team_id",
    "current_match_id",
    "current_date",
    "league_id",
    "season",
    "opponent_id_current",
    "result_current",
]

feature_cols = [c for c in df.columns if c not in context_cols]

print("Nº de features (histórico + agregados):", len(feature_cols))
print("Ligas disponíveis:", sorted(df["league_id"].unique()))

results = []         # métricas por liga
dfs_clusters = []    # para juntar os clusters de todas as ligas

for league_id, df_league in df.groupby("league_id"):
    league_name = LEAGUE_NAMES.get(league_id, str(league_id))
    n_rows = len(df_league)

    print("\n" + "#" * 80)
    print(f"Liga {league_id} - {league_name}")
    print(f"Nº de janelas nesta liga: {n_rows}")

    if n_rows < K * 10:
        print("Poucos exemplos para clustering, pulando...")
        continue

    X = df_league[feature_cols].values
    y_true = df_league["result_current"].values

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Avaliação com rótulo real
    sil_true = silhouette_score(X_scaled, y_true)
    ch_true = calinski_harabasz_score(X_scaled, y_true)
    db_true = davies_bouldin_score(X_scaled, y_true)

    print("\n=== Rótulo REAL (result_current) nesta liga ===")
    print("Distribuição de rótulos:")
    print(pd.Series(y_true).value_counts().sort_index())
    print(f"Silhouette (real): {sil_true:.4f}")
    print(f"Calinski-Harabasz (real): {ch_true:.4f}")
    print(f"Davies-Bouldin (real): {db_true:.4f}")

    # KMeans k=3
    print(f"\n=== KMeans k={K} nesta liga (sem usar rótulo no treino) ===")
    kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_scaled)

    sil_km = silhouette_score(X_scaled, cluster_labels)
    ch_km = calinski_harabasz_score(X_scaled, cluster_labels)
    db_km = davies_bouldin_score(X_scaled, cluster_labels)

    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)

    print(f"Silhouette (KMeans): {sil_km:.4f}")
    print(f"Calinski-Harabasz (KMeans): {ch_km:.4f}")
    print(f"Davies-Bouldin (KMeans): {db_km:.4f}")
    print(f"ARI (vs rótulo real): {ari:.4f}")
    print(f"NMI (vs rótulo real): {nmi:.4f}")

    # DataFrame com clusters desta liga
    df_league_tmp = df_league.copy()
    df_league_tmp["cluster_kmeans_3"] = cluster_labels
    dfs_clusters.append(df_league_tmp)

    print("\nDistribuição de clusters (KMeans k=3):")
    print(df_league_tmp["cluster_kmeans_3"].value_counts().sort_index())

    print("\nCrosstab clusters x resultado atual (contagem absoluta):")
    ct_abs = pd.crosstab(df_league_tmp["cluster_kmeans_3"], df_league_tmp["result_current"])
    print(ct_abs)

    print("\nCrosstab clusters x resultado atual (proporção por cluster):")
    ct_prop = pd.crosstab(
        df_league_tmp["cluster_kmeans_3"],
        df_league_tmp["result_current"],
        normalize="index"
    )
    print(ct_prop)

    # Guardar resumo numérico para tabela final
    results.append({
        "league_id": league_id,
        "league_name": league_name,
        "n_janelas": n_rows,
        "sil_real": sil_true,
        "ch_real": ch_true,
        "db_real": db_true,
        "sil_kmeans": sil_km,
        "ch_kmeans": ch_km,
        "db_kmeans": db_km,
        "ari": ari,
        "nmi": nmi,
    })

# 2. DataFrame resumo de métricas
df_results = pd.DataFrame(results)
print("\n" + "=" * 80)
print("RESUMO DAS MÉTRICAS POR LIGA (vetor reduzido + agregados)")
print(df_results.sort_values("league_id"))

# 3. Concatenar clusters de todas as ligas e salvar tudo no banco
conn = sqlite3.connect(DB_PATH)

# Tabela de métricas por liga
df_results.to_sql(
    "clustering_hist5_reduced_by_league_metrics",
    conn,
    if_exists="replace",
    index=False,
)

# Tabela com todas as janelas + cluster por liga
if dfs_clusters:
    df_all_clusters = pd.concat(dfs_clusters, ignore_index=True)
    df_all_clusters.to_sql(
        "team_windows_hist5_reduced_kmeans3_by_league",
        conn,
        if_exists="replace",
        index=False,
    )
    print(
        "\nTabela 'team_windows_hist5_reduced_kmeans3_by_league' "
        "salva com todas as janelas + cluster por liga."
    )

conn.close()
print("\nResumo de métricas salvo na tabela 'clustering_hist5_reduced_by_league_metrics'.")
print("Fim do clustering_hist5_reduced_by_league.py")
