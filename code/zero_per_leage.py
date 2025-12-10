import sqlite3
import pandas as pd

conn = sqlite3.connect("../data/database.sqlite")

df_tm = pd.read_sql_query("SELECT * FROM team_match_features;", conn)
conn.close()

check_cols = ["shots_for", "shots_on_for", "possession_for", "crosses_for", "corners_for"]

zero_rate = (
    df_tm.groupby("league_id")[check_cols]
    .apply(lambda g: (g == 0).mean())
    .sort_index()
)

print(zero_rate)
