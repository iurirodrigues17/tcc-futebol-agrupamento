# clustering_stats_3_4_5_cluster_profiles.py
#
# Lê a tabela team_windows_stats_3_4_5_kmeans3
# e calcula perfis médios dos clusters (global e por liga)

import sqlite3
import pandas as pd

DB_PATH = "../data/database.sqlite"
SOURCE_TABLE = "team_windows_stats_3_4_5_kmeans3"

# nome da coluna de cluster usada nos experimentos
CLUSTER_COL = "cluster_kmeans_3_stats"

# nome da coluna da liga
LEAGUE_COL = "league_id"

# Tabelas de saída (podem ser ajustadas se quiser)
TARGET_TABLE_CLUSTER_PROFILE = "cluster_profiles_stats_3_4_5_global"
TARGET_TABLE_CLUSTER_LEAGUE_PROFILE = "cluster_profiles_stats_3_4_5_by_league"

def main():
    print("DB_PATH:", DB_PATH)
    print("Lendo tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Shape da tabela de origem:", df.shape)
    print("Colunas disponíveis:", df.columns.tolist()[:20])

    if CLUSTER_COL not in df.columns:
        raise ValueError(f"Coluna de cluster '{CLUSTER_COL}' não encontrada na tabela {SOURCE_TABLE}.")

    # 1) Selecionar features de interesse (médias das últimas 5 partidas)
    features_last5 = [
        "mean_goals_for_last5",
        "mean_shots_for_last5",
        "mean_shots_on_for_last5",
        "mean_possession_for_last5",
        "mean_corners_for_last5",
        "mean_crosses_for_last5",
        "mean_fouls_for_last5",
        # se quiser incluir variabilidade também:
        "std_goals_for_last5",
        "std_shots_for_last5",
        "std_shots_on_for_last5",
        "std_possession_for_last5",
        "std_corners_for_last5",
        "std_crosses_for_last5",
        "std_fouls_for_last5",
    ]

    # Garante que só usamos colunas que realmente existem na tabela
    features_last5 = [c for c in features_last5 if c in df.columns]
    print("\nFeatures usadas para perfil (last5):")
    print(features_last5)

    # 2) Perfil global dos clusters (média das features por cluster)
    print("\nCalculando perfil global dos clusters (médias das últimas 5 partidas)...")

    df_profile_global = (
        df.groupby(CLUSTER_COL)[features_last5]
          .mean()
          .reset_index()
          .sort_values(CLUSTER_COL)
    )

    print("\nPerfil global dos clusters (primeiras linhas):")
    print(df_profile_global.head())

    # Salva no banco
    df_profile_global.to_sql(
        TARGET_TABLE_CLUSTER_PROFILE,
        conn,
        if_exists="replace",
        index=False,
    )
    print(f"\nTabela '{TARGET_TABLE_CLUSTER_PROFILE}' salva com sucesso no banco.")

    # 3) Perfil por liga + cluster (média das features por (liga, cluster))
    if LEAGUE_COL in df.columns:
        print("\nCalculando perfil por liga + cluster...")

        df_profile_by_league = (
            df.groupby([LEAGUE_COL, CLUSTER_COL])[features_last5]
              .mean()
              .reset_index()
              .sort_values([LEAGUE_COL, CLUSTER_COL])
        )

        print("\nPerfil por liga + cluster (primeiras linhas):")
        print(df_profile_by_league.head())

        df_profile_by_league.to_sql(
            TARGET_TABLE_CLUSTER_LEAGUE_PROFILE,
            conn,
            if_exists="replace",
            index=False,
        )
        print(f"\nTabela '{TARGET_TABLE_CLUSTER_LEAGUE_PROFILE}' salva com sucesso no banco.")
    else:
        print(f"\nColuna '{LEAGUE_COL}' não encontrada. Pulando perfil por liga + cluster.")

    conn.close()
    print("\n Fim do clustering_stats_3_4_5_cluster_profiles.py")

if __name__ == "__main__":
    main()
