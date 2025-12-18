# Lê team_windows_hist5_reduced, calcula mean/std para 3, 4 e 5 jogos
# e gera uma tabela no banco com:
# - contexto (time, partida atual, liga, etc.)
# - médias e desvios-padrão das últimas 5, 4 e 3 partidas
# - proporção de jogos em casa (home_ratio) para cada horizonte

import sqlite3
import pandas as pd

DB_PATH = "../data/database.sqlite"
SOURCE_TABLE = "team_windows_hist5_reduced"
TARGET_TABLE = "team_windows_stats_3_4_5"


def main():
    print("DB_PATH:", DB_PATH)
    print("Lendo tabela de origem:", SOURCE_TABLE)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE};", conn)

    print("Shape da tabela de origem:", df.shape)
    print("Colunas de exemplo:", df.columns[:20].tolist())

    # 1) Colunas de CONTEXTO (NÃO entram no vetor de features,
    # mas são importantes para análise e para ligar o vetor à partida)
    context_cols = [
        "team_id",
        "current_match_id",
        "current_date",
        "league_id",
        "season",
        "opponent_id_current",
        "result_current",  # rótulo real: -1, 0, 1
    ]

    # Garante que só são capturadas as existentes
    context_cols = [c for c in context_cols if c in df.columns]

    # Base do novo DataFrame
    df_stats = df[context_cols].copy()

    # 2) Atributos por partida a resumir por média/desvio
    per_match_attrs = [
        "goals_for",
        # "goals_against"
        "shots_for",
        "shots_on_for",
        "possession_for",
        "corners_for",
        "crosses_for",
        "fouls_for",
        "yellows_for",
        "reds_for",
    ]

    # is_home é tratado separado (vira proporção de jogos em casa)
    home_attr = "is_home"

    # 3) Para cada horizonte (5, 4, 3) é calculado média e desvio
    horizons = [5, 4, 3]

    for h in horizons:
        print(f"\nCalculando estatísticas para últimas {h} partidas...")

        # Médias e desvios dos atributos numéricos
        for attr in per_match_attrs:
            cols = [f"{attr}_{i}" for i in range(1, h + 1)]

            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"Aviso: pulando {attr} para horizonte {h}, faltam colunas:", missing)
                continue

            col_mean = f"mean_{attr}_last{h}"
            col_std = f"std_{attr}_last{h}"

            df_stats[col_mean] = df[cols].mean(axis=1)
            # std com ddof=0 (população) + NaN -> 0.0
            df_stats[col_std] = df[cols].std(axis=1, ddof=0).fillna(0.0)

        # Proporção de jogos em casa no horizonte (0 a 1)
        home_cols = [f"{home_attr}_{i}" for i in range(1, h + 1)]
        missing_home = [c for c in home_cols if c not in df.columns]
        if missing_home:
            print(f"Aviso: pulando home_ratio_last{h}, faltam colunas:", missing_home)
        else:
            df_stats[f"home_ratio_last{h}"] = df[home_cols].mean(axis=1)

    print(
        "\nShape do df_stats (contexto + médias/desvios + home_ratio 3/4/5):",
        df_stats.shape,
    )

    print("Algumas colunas criadas de exemplo:")
    exemplo_cols = [
        c
        for c in df_stats.columns
        if ("mean_goals_for" in c)
        or ("home_ratio" in c)
        or ("std_goals_for" in c)
    ][:10]
    print(exemplo_cols)

    # 4) Salvar a nova tabela no banco
    df_stats.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nTabela '{TARGET_TABLE}' criada/salva no banco com sucesso.")


if __name__ == "__main__":
    main()
