import sqlite3
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

DB_PATH = "../data/database.sqlite"
TABLE = "team_windows_5_strict"  # ou padded, se quiser comparar
WINDOW_SIZE = 5

# 1) Load
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f"SELECT * FROM {TABLE};", conn)
conn.close()

print("Tabela carregada:", df.shape)

# 2) Identificar colunas de features
context_cols = ["team_id", "ref_match_id", "ref_date", "league_id", "season", "opponent_id_ref"]
feature_cols = [c for c in df.columns if c not in context_cols]

print("Nº de features:", len(feature_cols))

# 3) Montar X
X = df[feature_cols].values

# 4) Padronização global
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5) Função de avaliação
def evaluate(labels, X_input):
    # Alguns algoritmos geram -1 como ruído (DBSCAN)
    unique = set(labels)
    if len(unique - {-1}) < 2:
        return {"silhouette": None, "ch": None, "db": None}

    # Para silhouette, remover ruído se existir
    mask = labels != -1
    X_eval = X_input[mask]
    y_eval = labels[mask]

    return {
        "silhouette": float(silhouette_score(X_eval, y_eval)),
        "ch": float(calinski_harabasz_score(X_eval, y_eval)),
        "db": float(davies_bouldin_score(X_eval, y_eval)),
    }

# 6) Testes com K variando (KMeans + Hierárquico + Birch)
results = []

for k in [3, 5, 7, 9]:
    # KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels_km = km.fit_predict(X_scaled)
    m = evaluate(labels_km, X_scaled)
    results.append({"alg": "KMeans", "k": k, **m})

    # Agglomerative
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_ag = ag.fit_predict(X_scaled)
    m = evaluate(labels_ag, X_scaled)
    results.append({"alg": "Agglomerative(ward)", "k": k, **m})

    # Birch
    br = Birch(n_clusters=k)
    labels_br = br.fit_predict(X_scaled)
    m = evaluate(labels_br, X_scaled)
    results.append({"alg": "Birch", "k": k, **m})

# 7) DBSCAN (sem k)
dbs = DBSCAN(eps=1.2, min_samples=10)
labels_dbs = dbs.fit_predict(X_scaled)
m = evaluate(labels_dbs, X_scaled)

results.append({"alg": "DBSCAN", "k": None, **m})

# 8) Tabela de resultados
df_res = pd.DataFrame(results)
print("\nResultados:")
print(df_res.sort_values(["alg", "k"], na_position="last"))

# 9) Escolher um modelo final (exemplo: KMeans com k=5)
final_k = 3
final_model = KMeans(n_clusters=final_k, random_state=42, n_init="auto")
df["cluster"] = final_model.fit_predict(X_scaled)

print(f"\nDistribuição de clusters (KMeans k={final_k}):")
print(df["cluster"].value_counts().sort_index())

# 10) Salvar resultado em uma nova tabela no SQLite
conn = sqlite3.connect(DB_PATH)
df.to_sql("team_windows_5_strict_clusters_kmeans_3", conn, if_exists="replace", index=False)
conn.close()

print(df["cluster"].value_counts().sort_index())
print(f"\nTabela com clusters salva: {TABLE}_clusters_kmeans_{final_k}")

context_cols = ["team_id", "ref_match_id", "ref_date", "league_id", "season", "opponent_id_ref"]
feature_cols = [c for c in df.columns if c not in context_cols + ["cluster"]]

def mean_group(prefix):
    cols = [c for c in feature_cols if c.startswith(prefix)]
    return df.groupby("cluster")[cols].mean().mean(axis=1)

summary = pd.DataFrame({
    "gols_pro": mean_group("goals_for"),
    "gols_contra": mean_group("goals_against"),
    "chutes_pro": mean_group("shots_for"),
    "chutes_no_alvo_pro": mean_group("shots_on_for"),
    "posse_pro": mean_group("possession_for"),
    "faltas_pro": mean_group("fouls_for"),
    "amarelos_pro": mean_group("yellows_for"),
    "vermelhos_pro": mean_group("reds_for"),
})

print("\nResumo interpretável por cluster (média nas 5 partidas):")
print(summary)

print("\nDistribuição por liga:")
print(df.groupby(["league_id", "cluster"]).size().unstack(fill_value=0))
