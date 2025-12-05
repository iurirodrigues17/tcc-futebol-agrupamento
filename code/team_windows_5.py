import sqlite3
import pandas as pd
import numpy as np

# Define o caminho do babnco e o tamanho da janela de partidas
DB_PATH = "../data/database.sqlite"
WINDOW_SIZE = 5

# As colunas numéricas por PARTIDA 
# Representando o "pacote" que vai ser repetido 5 vezes no vetor final.
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
    "is_home"
]

# Conecta ao banco
conn = sqlite3.connect(DB_PATH)

df_tm = pd.read_sql_query("SELECT * FROM team_match_features;", conn)
print("team_match_features:", df_tm.shape)

# Garantir tipos e limpeza básica
df_tm["date"] = pd.to_datetime(df_tm["date"], errors="coerce")

# Se houver NaNs em colunas numéricas, zerar para evitar quebra
df_tm[base_feature_cols] = df_tm[base_feature_cols].fillna(0.0)

# Constroi as janelas
all_windows = []
context_rows = []

# Ex.: goals_for_1, goals_for_2, ... goals_for_5
# Onde _1 = partida MAIS RECENTE da janela
window_feature_names = []
for k in range(1, WINDOW_SIZE + 1):
    for col in base_feature_cols:
        window_feature_names.append(f"{col}_{k}")

# Agrupar por time
for team_id, df_team in df_tm.groupby("team_id"):
    # Ordenar por data crescente (antiga -> recente)
    df_team = df_team.sort_values("date").reset_index(drop=True)

    if len(df_team) < WINDOW_SIZE:
        continue

    # i é o índice da partida que fecha a janela
    for i in range(WINDOW_SIZE - 1, len(df_team)):
        window = df_team.iloc[i - WINDOW_SIZE + 1: i + 1].copy()

        # Inverter para que a ordem interna do vetor siga:
        # _1 = última partida (mais recente)
        # _2 = penúltima...
        window = window.iloc[::-1].reset_index(drop=True)

        feats = []
        for _, row in window.iterrows():
            feats.extend(row[base_feature_cols].tolist())

        all_windows.append(feats)

        # Contexto da janela
        ref_row = df_team.iloc[i]
        context_rows.append({
            "team_id": int(team_id),
            "ref_match_id": int(ref_row["match_id"]),
            "ref_date": ref_row["date"],
            "league_id": int(ref_row["league_id"]) if "league_id" in ref_row else None,
            "season": ref_row["season"] if "season" in ref_row else None,
            "opponent_id_ref": int(ref_row["opponent_id"])
        })

# Converter em estruturas finais
X_windows = np.array(all_windows, dtype=float)
df_context = pd.DataFrame(context_rows)

print("Nº de janelas criadas:", len(df_context))
print("Formato de X_windows:", X_windows.shape)
print("Nº de features por janela esperado:",
      WINDOW_SIZE, "x", len(base_feature_cols), "=",
      WINDOW_SIZE * len(base_feature_cols))

df_windows = pd.concat(
    [df_context.reset_index(drop=True),
     pd.DataFrame(X_windows, columns=window_feature_names)],
    axis=1
)

print("df_windows:", df_windows.shape)
print(df_windows.head(3))

# Salvando
table_name = f"team_windows_{WINDOW_SIZE}"
df_windows.to_sql(table_name, conn, if_exists="replace", index=False)

print(f"\nTabela '{table_name}' criada/sobrescrita com sucesso no banco.")

conn.close()
