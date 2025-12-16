import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "../data/database.sqlite"
WINDOW_SIZE = 5

# Atributos por partida do ponto de vista do time (janela)
base_feature_cols = [
    "goals_for",
    "shots_for",
    "shots_on_for",
    "possession_for",
    "corners_for",
    "crosses_for",
    "fouls_for",
    "yellows_for",
    "reds_for",
    "is_home",      # 0 = casa, 1 = fora
    "result_code",  # -1, 0, 1 do jogo histórico
]

# Atributos agregados nas 5 partidas
agg_feature_cols = [
    "wins_last5",
    "draws_last5",
    "losses_last5",
    "points_last5",
    "goals_for_last5",
    "shots_for_last5",
    "shots_on_for_last5",
    "yellows_for_last5",
    "reds_for_last5",
    "avg_possession_for_last5",
]


def build_window_feature_names(window_size: int, cols: list[str]) -> list[str]:
    """
    Gera nomes <col>_k para cada posição da janela:
    - k = 1 -> partida MAIS recente do histórico (logo antes da atual)
    - k = window_size -> partida mais antiga.
    """
    names = []
    for k in range(1, window_size + 1):
        for col in cols:
            names.append(f"{col}_{k}")
    return names


def main():
    # 1. Conectar ao banco e carregar a base por time/partida
    conn = sqlite3.connect(DB_PATH)

    df_tm = pd.read_sql_query("SELECT * FROM team_match_features;", conn)
    print("team_match_features:", df_tm.shape)

    # Converter data
    df_tm["date"] = pd.to_datetime(df_tm["date"], errors="coerce")

    # Garantir que colunas usadas estão sem NaN
    df_tm[base_feature_cols] = df_tm[base_feature_cols].fillna(0.0)

    # Nomes das features da janela (histórico)
    window_feature_names = build_window_feature_names(WINDOW_SIZE, base_feature_cols)
    print("Nº de features por janela (histórico):", len(window_feature_names))
    print("Exemplos:", window_feature_names[:10])

    all_windows = []
    all_contexts = []

    # 2. Geração de janelas por time
    for team_id, df_team in df_tm.groupby("team_id"):
        df_team = df_team.sort_values("date").reset_index(drop=True)
        n = len(df_team)

        # Precisa ter pelo menos 5 partidas anteriores + 1 atual
        if n <= WINDOW_SIZE:
            continue

        for i in range(WINDOW_SIZE, n):
            # Histórico: 5 partidas anteriores (i-5 ... i-1)
            hist = df_team.iloc[i - WINDOW_SIZE:i].copy()

            # Reordena para:
            #   _1 -> partida mais recente (i-1)
            #   _5 -> partida mais antiga (i-5)
            hist = hist.iloc[::-1].reset_index(drop=True)

            # ---- Parte 1: concatenar atributos partida a partida ----
            feats = []
            for _, row in hist.iterrows():
                feats.extend(row[base_feature_cols].tolist())

            # ---- Parte 2: atributos agregados das 5 partidas ----
            results_hist = hist["result_code"].values

            wins_5 = np.sum(results_hist == 1)
            draws_5 = np.sum(results_hist == 0)
            losses_5 = np.sum(results_hist == -1)
            points_5 = 3 * wins_5 + draws_5

            goals_for_5 = hist["goals_for"].sum()
            shots_for_5 = hist["shots_for"].sum()
            shots_on_for_5 = hist["shots_on_for"].sum()
            yellows_for_5 = hist["yellows_for"].sum()
            reds_for_5 = hist["reds_for"].sum()
            avg_possession_for_5 = hist["possession_for"].mean()

            agg_feats = [
                wins_5,              # wins_last5
                draws_5,             # draws_last5
                losses_5,            # losses_last5
                points_5,            # points_last5
                goals_for_5,         # goals_for_last5
                shots_for_5,         # shots_for_last5
                shots_on_for_5,      # shots_on_for_last5
                yellows_for_5,       # yellows_for_last5
                reds_for_5,          # reds_for_last5
                avg_possession_for_5 # avg_possession_for_last5
            ]


            feats.extend(agg_feats)

            all_windows.append(feats)

            # Contexto + rótulo da partida atual
            current_row = df_team.iloc[i]
            all_contexts.append({
                "team_id": int(current_row["team_id"]),
                "current_match_id": int(current_row["match_id"]),
                "current_date": current_row["date"],
                "league_id": int(current_row["league_id"]),
                "season": current_row["season"],
                "opponent_id_current": int(current_row["opponent_id"]),
                "result_current": int(current_row["result_code"]),  # -1,0,1 da atual
            })

    # 3. Montar DataFrame final
    X = np.array(all_windows, dtype=float)
    df_context = pd.DataFrame(all_contexts)

    print("Nº de janelas geradas:", len(df_context))
    print("Formato de X (janelas, features):", X.shape)

    all_feature_names = window_feature_names + agg_feature_cols
    print("Total de features por janela:", len(all_feature_names))

    df_windows = pd.concat(
        [df_context.reset_index(drop=True),
         pd.DataFrame(X, columns=all_feature_names)],
        axis=1
    )

    print("df_windows_hist5_reduced:", df_windows.shape)
    print(df_windows.head(3))

    # 4. Salvar no banco
    table_name = "team_windows_hist5_reduced"
    df_windows.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Tabela '{table_name}' criada/salva no banco.")

    conn.close()
    print("Conexão fechada.")


if __name__ == "__main__":
    main()
