# 1. Imports
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 2. Conectar ao banco e carregar a tabela de features
conn = sqlite3.connect("database.sqlite")

# Lê toda a tabela match_features criada
df = pd.read_sql_query("SELECT * FROM match_features;", conn)

print("Primeiras linhas da match_features:")
print(df.head())
print("Formato do df (linhas, colunas):", df.shape)

# 3. Definir colunas de features (HOME/AWAY separados) e de contexto
feature_cols = [
    "shots_home", "shots_away",
    "shots_on_home", "shots_on_away",
    "possession_home", "possession_away",
    "corners_home", "corners_away",
    "crosses_home", "crosses_away",
    "fouls_home", "fouls_away",
    "yellows_home", "yellows_away",
    "reds_home", "reds_away",
]

# Colunas de contexto (não entram no vetor, mas estão aqui e podem auxiliar na posterior análise)
context_cols = [
    "match_id",
    "league_id",
    "season",
    "home_team_api_id",
    "away_team_api_id",
    "home_team_goal",
    "away_team_goal",
    "goal_diff",
    "result_label",
]

# 4. Trata valores ausentes (NaN) nas features
# Alguns jogos podem não ter posse ou certos eventos registrados, logo preenchem NaN com 0.0 nas features.
df[feature_cols] = df[feature_cols].fillna(0.0)

print("\nVerificando se ainda há NaN nas features:")
print(df[feature_cols].isna().sum())  # ideal é imprimir tudo 0

# 5. Padronizar as features por liga (z-score)
# Cria um novo DataFrame df_scaled, onde cada feature é padronizada
# DENTRO de cada liga. Assim, por exemplo, shots_home_z indica
# quantos desvios-padrão acima/abaixo da média da liga aquele valor está.
dfs_scaled = []

for league_id, df_league in df.groupby("league_id"):
    print(f"\nPadronizando liga {league_id} com {len(df_league)} partidas...")
    
    scaler = StandardScaler()
    X = df_league[feature_cols].values  # Matriz de features dessa liga
    
    # fit_transform: calcula média/desvio na liga e aplica o z-score
    X_scaled = scaler.fit_transform(X)
    
    # Copia o df dessa liga e adiciona colunas *_z com os valores padronizados
    df_league_scaled = df_league.copy()
    for i, col in enumerate(feature_cols):
        df_league_scaled[col + "_z"] = X_scaled[:, i]
    
    dfs_scaled.append(df_league_scaled)

# Junta todas as ligas de volta num único DataFrame
df_scaled = pd.concat(dfs_scaled, ignore_index=True)

# Lista das colunas padronizadas (que vão compor o vetor)
z_feature_cols = [col + "_z" for col in feature_cols]

print("\nPrimeiras linhas com features padronizadas (z-score):")
print(df_scaled[context_cols + z_feature_cols].head())

# 6. Monta a matriz de vetores de atributos (X_vectors)
# Aqui, cada linha de X_vectors é um VETOR de atributos de uma partida,
# com as estatísticas separadas de mandante e visitante, padronizadas por liga.
X_vectors = df_scaled[z_feature_cols].values

print("\nFormato de X_vectors (n_partidas, n_features):", X_vectors.shape)

print("\nPrimeiro vetor de atributos (array):")
print(X_vectors[0])

print("\nPrimeiro vetor de atributos com nomes das features:")
print(df_scaled[z_feature_cols].iloc[0])

# Fecha a conexão com o banco
conn.close()