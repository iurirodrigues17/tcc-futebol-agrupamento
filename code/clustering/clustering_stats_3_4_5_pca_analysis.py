# clustering_stats_3_4_5_pca_analysis.py
#
# Análise com PCA dos vetores de desempenho (3/4/5),
# usando a tabela com clusters k=3 (vitória/empate/derrota).
#
# - Lê a tabela team_windows_stats_3_4_5_kmeans3
# - Seleciona as colunas mean_/std_
# - Padroniza os dados
# - Aplica PCA
# - Imprime variância explicada das componentes
# - Calcula uma medida de "importância global" dos atributos
#   (soma dos loadings^2 ponderada pela variância explicada)
# - Salva:
#   * Tabela de importâncias dos atributos em: clustering_stats_3_4_5_pca_feature_importance
#   * Scores das PCs + rótulos em: clustering_stats_3_4_5_pca_scores

import sqlite3
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DB_PATH = "../data/database.sqlite"
SOURCE_TABLE = "team_windows_stats_3_4_5_kmeans3"

# Configuração de quais famílias de atributos usar (como no experimento D)
USE_FOULS = True
USE_YELLOWS = True
USE_REDS = True

# Quantas componentes principais vamos analisar/salvar
N_COMPONENTS = 10  # pode ajustar se quiser


def main():
    print("DB_PATH:", DB_PATH)
    print("Lendo tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)

    # Lê a tabela com os clusters já calculados
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)
    print("Shape da tabela de origem:", df.shape)
    print("Colunas disponíveis:", df.columns[:20].tolist())
    
    #Filtro de ligas
    EXCLUDED_LEAGUES = [24558]

    before_filter = len(df)
    df = df[~df["league_id"].isin(EXCLUDED_LEAGUES)].copy()
    after_filter = len(df)

    print("\n=== Filtro de ligas ===")
    print(f"Ligas excluídas: {EXCLUDED_LEAGUES}")
    print(f"Registros removidos pelo filtro de ligas: {before_filter - after_filter}")
    print(f"Shape antes do filtro: ({before_filter}, {df.shape[1]})")
    print(f"Shape após o filtro: ({after_filter}, {df.shape[1]})\n")

    # 1) Selecionar colunas de features (mean_ / std_), como no experimento D
    all_feat_cols = [c for c in df.columns if c.startswith("mean_") or c.startswith("std_")]
    all_feat_cols = sorted(all_feat_cols)
    print(f"\nTotal de colunas mean_/std_ encontradas: {len(all_feat_cols)}")

    # Famílias que podemos desligar (faltas, amarelos, vermelhos)
    foul_prefixes = ["_fouls_for_last3", "_fouls_for_last4", "_fouls_for_last5"]
    yellow_prefixes = ["_yellows_for_last3", "_yellows_for_last4", "_yellows_for_last5"]
    red_prefixes = ["_reds_for_last3", "_reds_for_last4", "_reds_for_last5"]

    def is_family(col: str, base: str, suffixes):
        return any(col.endswith(base + suf) for suf in suffixes)

    selected_cols = []
    removed_fouls = []
    removed_yellows = []
    removed_reds = []

    for c in all_feat_cols:
        if not USE_FOULS and is_family(c.replace("mean", "").replace("std", ""), "fouls", foul_prefixes):
            removed_fouls.append(c)
            continue
        if not USE_YELLOWS and is_family(c.replace("mean", "").replace("std", ""), "yellows", yellow_prefixes):
            removed_yellows.append(c)
            continue
        if not USE_REDS and is_family(c.replace("mean", "").replace("std", ""), "reds", red_prefixes):
            removed_reds.append(c)
            continue
        # caso contrário, mantemos
        selected_cols.append(c)

    print("\nResumo da seleção de features (para PCA):")
    print(f"  Total original (mean_/std_): {len(all_feat_cols)}")
    print(f"  Usando faltas?      {USE_FOULS}  -> removidas: {len(removed_fouls)}")
    print(f"  Usando amarelos?    {USE_YELLOWS} -> removidas: {len(removed_yellows)}")
    print(f"  Usando vermelhos?   {USE_REDS}    -> removidas: {len(removed_reds)}")
    print(f"  Nº de features selecionadas: {len(selected_cols)}")
    print("  Exemplos de features selecionadas:", selected_cols[:15])

    # 2) Montar matriz X
    X = df[selected_cols].copy()

    # Tratamento simples de NaN (se tiver): substitui pela média da coluna
    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        print(f"\nAviso: há {n_missing} valores NaN em X. Substituindo pela média da coluna...")
        X = X.fillna(X.mean())

    # 3) Padroniza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Shape de X_scaled:", X_scaled.shape)

    # 4) PCA
    n_comp = min(N_COMPONENTS, X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    print("\n=== Variância explicada pelas componentes principais ===")
    for i, (vr, cv) in enumerate(zip(var_ratio, cum_var_ratio), start=1):
        print(f"PC{i:02d}: var={vr:.4f}  (acumulada={cv:.4f})")

    # 5) Importância dos atributos
    # loadings: [n_features x n_components]
    loadings = pca.components_.T
    # importância global: soma_j (loading_ij^2 * var_ratio_j)
    importance = (loadings**2 * var_ratio).sum(axis=1)

    df_importance = pd.DataFrame({
        "feature": selected_cols,
        "importance_pca": importance
    })

    # Também podemos guardar alguns detalhes de loadings nas primeiras PCs
    for k in range(min(3, n_comp)):
        df_importance[f"abs_loading_PC{k+1}"] = np.abs(loadings[:, k])

    df_importance = df_importance.sort_values("importance_pca", ascending=False).reset_index(drop=True)

    print("\n=== Top 20 atributos mais importantes segundo o PCA ===")
    print(df_importance.head(20))

    # Salvar tabela de importâncias no banco
    df_importance.to_sql(
        "clustering_stats_3_4_5_pca_feature_importance",
        conn,
        if_exists="replace",
        index=False
    )
    print("\nTabela 'clustering_stats_3_4_5_pca_feature_importance' salva com sucesso no banco.")

    # 6) Salvar scores das PCs + rótulos e clusters para facilitar gráficos
    #   Incluímos até N_COMPONENTS PCs
    pc_cols = [f"PC{i+1}" for i in range(n_comp)]
    df_scores = pd.DataFrame(X_pca[:, :n_comp], columns=pc_cols)

    # Rótulos que queremos anexar
    label_cols = []
    for col in ["result_current", "cluster_kmeans_3_stats", "league_id", "team_id", "current_match_id"]:
        if col in df.columns:
            label_cols.append(col)

    df_scores = pd.concat([df[label_cols].reset_index(drop=True), df_scores], axis=1)

    print("\nExemplo de linhas dos scores PCA com rótulos:")
    print(df_scores.head())

    df_scores.to_sql(
        "clustering_stats_3_4_5_pca_scores",
        conn,
        if_exists="replace",
        index=False
    )
    print("\nTabela 'clustering_stats_3_4_5_pca_scores' salva com sucesso no banco.")

    conn.close()
    print("\n Fim do clustering_stats_3_4_5_pca_analysis.py")


if __name__ == "__main__":
    main()
