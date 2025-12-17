# Cada linha da team_match_features representa o desempenho de UM time em UMA partida.

import sqlite3
import pandas as pd

# Caminho do banco
DB_PATH = "../data/database.sqlite"

# 1. Conectar ao banco
conn = sqlite3.connect(DB_PATH)

# 2. Carregar a tabela de partidas (Match)
df_matches = pd.read_sql_query(
    """
    SELECT
        id AS match_id,
        date
    FROM Match;
    """,
    conn
)
print("df_matches:", df_matches.shape)

# 3. Carregar a tabela match_features
df_feat = pd.read_sql_query("SELECT * FROM match_features;", conn)
print("df_feat:", df_feat.shape)

# 4. Juntar Match + match_features pela chave match_id
df = df_matches.merge(df_feat, on="match_id", how="inner")

print("df (Match + match_features):", df.shape)
print(df.head())

# Conferir colunas
print("\nColunas de df:")
print(df.columns)

# 5. Construir visão do ponto de vista do time mandante
home = pd.DataFrame({
    "match_id": df["match_id"],
    "date": df["date"],
    "league_id": df["league_id"],
    "season": df["season"],

    "team_id": df["home_team_api_id"],
    "opponent_id": df["away_team_api_id"],
    "is_home": 0,  # 0 = mandante

    "goals_for": df["home_team_goal"],
    "goals_against": df["away_team_goal"],

    "shots_for": df["shots_home"],
    "shots_against": df["shots_away"],

    "shots_on_for": df["shots_on_home"],
    "shots_on_against": df["shots_on_away"],

    "possession_for": df["possession_home"],
    "possession_against": df["possession_away"],

    "corners_for": df["corners_home"],
    "corners_against": df["corners_away"],

    "crosses_for": df["crosses_home"],
    "crosses_against": df["crosses_away"],

    "fouls_for": df["fouls_home"],
    "fouls_against": df["fouls_away"],

    "yellows_for": df["yellows_home"],
    "yellows_against": df["yellows_away"],

    "reds_for": df["reds_home"],
    "reds_against": df["reds_away"],
})

# 6. Construir visão do ponto de vista do time visitante
away = pd.DataFrame({
    "match_id": df["match_id"],
    "date": df["date"],
    "league_id": df["league_id"],
    "season": df["season"],

    "team_id": df["away_team_api_id"],
    "opponent_id": df["home_team_api_id"],
    "is_home": 1,  # 1 = visitante (fora de casa)

    "goals_for": df["away_team_goal"],
    "goals_against": df["home_team_goal"],

    "shots_for": df["shots_away"],
    "shots_against": df["shots_home"],

    "shots_on_for": df["shots_on_away"],
    "shots_on_against": df["shots_on_home"],

    "possession_for": df["possession_away"],
    "possession_against": df["possession_home"],

    "corners_for": df["corners_away"],
    "corners_against": df["corners_home"],

    "crosses_for": df["crosses_away"],
    "crosses_against": df["crosses_home"],

    "fouls_for": df["fouls_away"],
    "fouls_against": df["fouls_home"],

    "yellows_for": df["yellows_away"],
    "yellows_against": df["yellows_home"],

    "reds_for": df["reds_away"],
    "reds_against": df["reds_home"],
})

# 7. Unir mandante + visitante em uma única tabela "por time e partida"
team_match = pd.concat([home, away], ignore_index=True)

print("\nteam_match (linhas, colunas):", team_match.shape)
print(team_match.head())

# 8. Tratar NaNs nas colunas numéricas (se houver)
num_cols = [
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_on_for", "shots_on_against",
    "possession_for", "possession_against",
    "corners_for", "corners_against",
    "crosses_for", "crosses_against",
    "fouls_for", "fouls_against",
    "yellows_for", "yellows_against",
    "reds_for", "reds_against",
]
team_match[num_cols] = team_match[num_cols].fillna(0.0)

# 9. Salvar no banco como nova tabela
team_match.to_sql("team_match_features", conn, if_exists="replace", index=False)

print("\nTabela 'team_match_features' criada/sobrescrita com sucesso no banco.")

conn.close()
