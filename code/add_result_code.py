import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "../data/database.sqlite"

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query("SELECT * FROM team_match_features;", conn)
print("team_match_features antes:", df.shape)

# Cria o código de resultado do ponto de vista do time
df["result_code"] = np.where(
    df["goals_for"] > df["goals_against"],  # vitória
    1,
    np.where(
        df["goals_for"] < df["goals_against"],  # derrota
        -1,
        0  # empate
    )
)

print(df[["goals_for", "goals_against", "result_code"]].head(10))

df.to_sql("team_match_features", conn, if_exists="replace", index=False)
print("Atualizado team_match_features com result_code.")

conn.close()
