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
TABLE = "team_windows_hist5"   # tabela das janelas: 5 partidas anteriores + resultado atual
K = 3                          # 3 grupos, pois temos vitória / empate / derrota

EXCLUDED_LEAGUES = [24558]

# 1. Carrega os dados
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

# Colunas de contexto (não entram em X)
context_cols = [
    "team_id",
    "current_match_id",
    "current_date",
    "league_id",
    "season",
    "opponent_id_current",
    "result_current",   # rótulo real da partida atual (−1, 0, 1)
]

feature_cols = [c for c in df.columns if c not in context_cols]

print("Nº de features (histórico 5 partidas):", len(feature_cols))

X = df[feature_cols].values
y_true = df["result_current"].values

print("\nDistribuição de rótulos (result_current):")
print(pd.Series(y_true).value_counts().sort_index())  # -1, 0, 1

# 2. Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Avaliação da separação real (usando o rótulo)
# Aqui NÃO tem clustering: usamos diretamente vitória/empate/derrota como "agrupamento real"
# As janelas de vitórias, empates e derrotas separam bem no espaço de atributos?

sil_true = silhouette_score(X_scaled, y_true)
ch_true = calinski_harabasz_score(X_scaled, y_true)
db_true = davies_bouldin_score(X_scaled, y_true)

print("\n=== Separação usando o rótulo REAL (result_current) ===")
print(f"Silhouette (rótulo real): {sil_true:.4f}")
print(f"Calinski-Harabasz (rótulo real): {ch_true:.4f}")
print(f"Davies-Bouldin (rótulo real): {db_true:.4f}")

# 4. Agrupamento (KMeans) e comparação
print(f"\n=== KMeans com k={K} (sem usar rótulo no treino) ===")

kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(X_scaled)

# Métricas internas
sil_km = silhouette_score(X_scaled, cluster_labels)
ch_km = calinski_harabasz_score(X_scaled, cluster_labels)
db_km = davies_bouldin_score(X_scaled, cluster_labels)

# Métricas externas (comparando cluster vs rótulo real)
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)

print(f"Silhouette (KMeans): {sil_km:.4f}")
print(f"Calinski-Harabasz (KMeans): {ch_km:.4f}")
print(f"Davies-Bouldin (KMeans): {db_km:.4f}")
print(f"Adjusted Rand Index (vs rótulo real): {ari:.4f}")
print(f"NMI (vs rótulo real): {nmi:.4f}")

# 5. Pureza dos clusters
df["cluster_kmeans_3"] = cluster_labels

print("\nDistribuição de clusters (KMeans k=3):")
print(df["cluster_kmeans_3"].value_counts().sort_index())

print("\nCrosstab clusters x resultado atual (contagem absoluta):")
ct_abs = pd.crosstab(df["cluster_kmeans_3"], df["result_current"])
print(ct_abs)

print("\nCrosstab clusters x resultado atual (proporção por cluster):")
ct_prop = pd.crosstab(df["cluster_kmeans_3"], df["result_current"], normalize="index")
print(ct_prop)

# 6. Salva resultados no banco
conn = sqlite3.connect(DB_PATH)
df.to_sql(f"{TABLE}_kmeans3", conn, if_exists="replace", index=False)
conn.close()

print(f"\nTabela com clusters salva como: {TABLE}_kmeans3")
