import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "../data/database.sqlite"
WINDOW_SIZE = 5

# Atributos por partida do ponto de vista do time
base_feature_cols = [
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_on_for", "shots_on_against",
    "possession_for", "possession_against",
    "corners_for", "corners_against",
    "crosses_for", "crosses_against",
    "fouls_for", "fouls_against",
    "yellows_for", "yellows_against",
    "reds_for", "reds_against",
    "is_home",
    "result_code",  # resultado da partida histórica
]

def build_window_feature_names(window_size: int, cols: list[str]) -> list[str]:
    names = []
    for k in range(1, window_size + 1):
        for col in cols:
            names.append(f"{col}_{k}")
    return names

conn = sqlite3.connect(DB_PATH)
df_tm = pd.read_sql_query("SELECT * FROM team_match_features;", conn)

print("team_match_features:", df_tm.shape)

# Garantir tipos e NAs
df_tm["date"] = pd.to_datetime(df_tm["date"], errors="coerce")
df_tm[base_feature_cols] = df_tm[base_feature_cols].fillna(0.0)

window_feature_names = build_window_feature_names(WINDOW_SIZE, base_feature_cols)

all_windows = []
all_contexts = []

for team_id, df_team in df_tm.groupby("team_id"):
    df_team = df_team.sort_values("date").reset_index(drop=True)
    n = len(df_team)

    # è necessário de pelo menos 5 partidas ANTERIORES + 1 atual
    if n <= WINDOW_SIZE:
        continue

    # i = índice da PARTIDA ATUAL
    # janelas vão de i = WINDOW_SIZE até n-1
    for i in range(WINDOW_SIZE, n):
        # Histórico: 5 partidas anteriores -> [i-5, ..., i-1]
        hist = df_team.iloc[i - WINDOW_SIZE:i].copy()

        # ordenar dentro da janela pra manter convenção:
        # _1 = partida MAIS recente do histórico (i-1)
        # _5 = partida mais antiga (i-5)
        hist = hist.iloc[::-1].reset_index(drop=True)

        feats = []
        for _, row in hist.iterrows():
            feats.extend(row[base_feature_cols].tolist())

        all_windows.append(feats)

        current_row = df_team.iloc[i]
        all_contexts.append({
            "team_id": int(current_row["team_id"]),
            "current_match_id": int(current_row["match_id"]),
            "current_date": current_row["date"],
            "league_id": int(current_row["league_id"]),
            "season": current_row["season"],
            "opponent_id_current": int(current_row["opponent_id"]),
            # rótulo da part. atual
            "result_current": int(current_row["result_code"]),
        })

X = np.array(all_windows, dtype=float)
df_context = pd.DataFrame(all_contexts)

print("Nº de janelas geradas:", len(df_context))
print("Formato de X (janelas, features):", X.shape)
print("Esperado:", len(df_tm), "partidas por time com janelas = somatório de max(0, n_t - 5)")

df_windows = pd.concat(
    [df_context.reset_index(drop=True),
     pd.DataFrame(X, columns=window_feature_names)],
    axis=1
)

print("df_windows_hist5:", df_windows.shape)
print(df_windows.head(3))

table_name = "team_windows_hist5"
df_windows.to_sql(table_name, conn, if_exists="replace", index=False)
print(f"Tabela '{table_name}' criada/salva no banco.")

conn.close()
