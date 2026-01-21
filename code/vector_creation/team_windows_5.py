import sqlite3
import pandas as pd
import numpy as np

# Define o caminho do babnco e o tamanho da janela de partidas
DB_PATH = "../data/database.sqlite"
WINDOW_SIZE = 5

# Atributos numéricos por PARTIDA (perspectiva do time)
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

def build_window_feature_names(window_size: int, cols: list[str]) -> list[str]:
    names = []
    for k in range(1, window_size + 1):
        for col in cols:
            names.append(f"{col}_{k}")
    return names

def build_windows_for_team(df_team: pd.DataFrame, window_size: int, pad_missing: bool):
    """
    Gera janelas para um único time.
    - pad_missing=False: só gera janelas completas (strict)
    - pad_missing=True : gera uma janela por partida, preenchendo jogos ausentes com zeros (padded)
    """
    df_team = df_team.sort_values("date").reset_index(drop=True)

    windows = []
    contexts = []

    n = len(df_team)
    if n == 0:
        return windows, contexts

    # Índices finais das janelas:
    # strict: começa em window_size-1
    # padded: começa em 0 (uma janela para cada partida)
    start_i = 0 if pad_missing else (window_size - 1)

    for i in range(start_i, n):
        # pega o histórico disponível até i (inclusive)
        window = df_team.iloc[max(0, i - window_size + 1): i + 1].copy()

        # Se for strict e não tiver tamanho completo, pula
        if (not pad_missing) and (len(window) < window_size):
            continue

        # No modo padded: se faltam partidas, adicionar linhas zero
        if pad_missing and len(window) < window_size:
            missing = window_size - len(window)
            zero_rows = pd.DataFrame([{col: 0.0 for col in base_feature_cols}] * missing)
            # concatenar zeros ANTES do histórico real
            # (porque na ordem cronológica, "faltam jogos mais antigos")
            window_feats = pd.concat([zero_rows, window[base_feature_cols]], ignore_index=True)
        else:
            window_feats = window[base_feature_cols].copy()

        window_feats = window_feats.iloc[::-1].reset_index(drop=True)

        feats = []
        for _, row in window_feats.iterrows():
            feats.extend(row.tolist())

        windows.append(feats)

        # Contexto sempre baseado na partida de referência (a i-ésima)
        ref_row = df_team.iloc[i]
        contexts.append({
            "team_id": int(ref_row["team_id"]),
            "ref_match_id": int(ref_row["match_id"]),
            "ref_date": ref_row["date"],
            "league_id": int(ref_row["league_id"]),
            "season": ref_row["season"],
            "opponent_id_ref": int(ref_row["opponent_id"])
        })

    return windows, contexts

def build_all_windows(df_tm: pd.DataFrame, window_size: int, pad_missing: bool):
    all_windows = []
    all_contexts = []

    # assegurar tipos
    df_tm["date"] = pd.to_datetime(df_tm["date"], errors="coerce")
    df_tm[base_feature_cols] = df_tm[base_feature_cols].fillna(0.0)

    for team_id, df_team in df_tm.groupby("team_id"):
        w, c = build_windows_for_team(df_team, window_size, pad_missing)
        all_windows.extend(w)
        all_contexts.extend(c)

    X = np.array(all_windows, dtype=float) if all_windows else np.empty((0, window_size * len(base_feature_cols)))
    df_context = pd.DataFrame(all_contexts)
    return X, df_context

def expected_strict_windows_count(df_tm: pd.DataFrame, window_size: int) -> int:
    """
    Soma, para cada time, max(0, n - window_size + 1)
    """
    counts = df_tm.groupby("team_id").size()
    return int(((counts - window_size + 1).clip(lower=0)).sum())

# Conecta ao banco
conn = sqlite3.connect(DB_PATH)

df_tm = pd.read_sql_query("SELECT * FROM team_match_features;", conn)
print("team_match_features:", df_tm.shape)

# STRICT (sem zerar)
print("\nGerando janelas STRICT (sem preencher zero)...")

X_strict, ctx_strict = build_all_windows(df_tm, WINDOW_SIZE, pad_missing=False)
names = build_window_feature_names(WINDOW_SIZE, base_feature_cols)

df_strict = pd.concat(
    [ctx_strict.reset_index(drop=True),
     pd.DataFrame(X_strict, columns=names)],
    axis=1
)

exp_strict = expected_strict_windows_count(df_tm, WINDOW_SIZE)

print("Nº de janelas STRICT geradas:", len(df_strict))
print("Nº de janelas STRICT esperado:", exp_strict)
print("Formato X_strict:", X_strict.shape)

# Salvar
table_strict = f"team_windows_{WINDOW_SIZE}_strict"
df_strict.to_sql(table_strict, conn, if_exists="replace", index=False)
print(f"Tabela '{table_strict}' salva no banco.")

# PADDED (com zerar)
print("\nGerando janelas PADDED (preenchendo histórico faltante com zero)...")

X_pad, ctx_pad = build_all_windows(df_tm, WINDOW_SIZE, pad_missing=True)

df_pad = pd.concat(
    [ctx_pad.reset_index(drop=True),
     pd.DataFrame(X_pad, columns=names)],
    axis=1
)

print("Nº de janelas PADDED geradas:", len(df_pad))
print("Formato X_pad:", X_pad.shape)

# Salvar
table_pad = f"team_windows_{WINDOW_SIZE}_padded"
df_pad.to_sql(table_pad, conn, if_exists="replace", index=False)
print(f"Tabela '{table_pad}' salva no banco.")

conn.close()

print("\n Concluído.")
print("Tabelas geradas:")
print("-", table_strict)
print("-", table_pad)
