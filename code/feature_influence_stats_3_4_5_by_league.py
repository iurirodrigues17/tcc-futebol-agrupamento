import sqlite3
import pandas as pd

DB_PATH = "../data/database.sqlite"
SOURCE_TABLE = "team_windows_stats_3_4_5_kmeans3_by_league"
TARGET_TABLE = "feature_influence_stats_3_4_5_by_league"


def main():
    print("DB_PATH:", DB_PATH)
    print("Tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df_all = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Tabela carregada:", df_all.shape)
    print("Colunas de exemplo:", df_all.columns[:20].tolist())

    # Conferências básicas
    if "league_id" not in df_all.columns:
        raise ValueError("Coluna 'league_id' não encontrada na tabela de origem.")
    if "cluster_kmeans_3_stats" not in df_all.columns:
        raise ValueError("Coluna 'cluster_kmeans_3_stats' não encontrada. "
                         "Certifique-se de ter rodado clustering_stats_3_4_5_by_league.py.")

    leagues = sorted(df_all["league_id"].unique())
    print("Ligas disponíveis:", leagues)

    # Vamos acumular resultados em uma lista de dicts
    rows = []

    for league in leagues:
        print("\n" + "#" * 80)
        print(f"Analisando liga {league}")

        df = df_all[df_all["league_id"] == league].copy()
        n_rows = len(df)
        print(f"Nº de janelas nesta liga: {n_rows}")

        # Checagem básica
        n_clusters = df["cluster_kmeans_3_stats"].nunique()
        print(f"Nº de clusters encontrados nesta liga: {n_clusters}")
        if n_clusters < 2:
            print("Menos de 2 clusters nesta liga, pulando análise de variância entre centróides.")
            continue

        # Colunas de contexto (NÃO entram como features)
        context_cols = [
            "team_id",
            "current_match_id",
            "current_date",
            "league_id",
            "season",
            "opponent_id_current",
            "result_current",
            "cluster_kmeans_3_stats",
        ]
        context_cols = [c for c in context_cols if c in df.columns]

        # Features = tudo que não é contexto
        feature_cols = [c for c in df.columns if c not in context_cols]

        print(f"Nº de features para análise nesta liga: {len(feature_cols)}")

        # 1) Calcula centróides: média de cada feature por cluster
        centroids = df.groupby("cluster_kmeans_3_stats")[feature_cols].mean()

        # 2) Variância entre centróides para cada feature
        #    Quanto maior a variância, mais a feature "diferencia" os clusters.
        var_between_centroids = centroids.var(axis=0)

        # Ordena features por variância (descendente)
        var_sorted = var_between_centroids.sort_values(ascending=False)

        print("\nTop 10 atributos com maior variância entre centróides (liga", league, "):")
        print(var_sorted.head(10))

        # Guarda todas as features desta liga, com rank
        for rank, (feat, var_val) in enumerate(var_sorted.items(), start=1):
            rows.append({
                "league_id": league,
                "feature": feat,
                "var_between_centroids": float(var_val),
                "rank_in_league": rank,
                "n_janelas": n_rows,
                "n_clusters": n_clusters,
            })

    # Monta DataFrame final com todas as ligas
    if rows:
        df_out = pd.DataFrame(rows)
        print("\nResumo das top features por liga (primeiras linhas):")
        print(df_out.head(20))

        # Salva no banco
        df_out.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        print(f"\nTabela '{TARGET_TABLE}' salva com sucesso no banco.")
    else:
        print("\nNenhum resultado gerado. Verifique se há dados e clusters válidos nas ligas.")

    conn.close()
    print("Fim do feature_influence_stats_3_4_5_by_league.py")


if __name__ == "__main__":
    main()
