import pandas as pd

COLUMNS_NAME = ['identificativo_calciatore', 'nome_calciatore', 'tackle_vinti']


def make_list(list_dict, key='id'):
    if len(list_dict) == 0:
        return []
    return [i[key] for i in list_dict]


# IMPORT DATA
data = pd.read_json('worldCup.json')
players = pd.read_json('player.json')

# FROM LIST OF DICT TO LIST
data.tags = data.tags.apply(make_list)

# TACKELS
tackles = data[data.subEventId == 12]

# COUNT TACKLES BY PLAYER
tck_by_player = tackles.groupby(['playerId']).size().reset_index(name='n_tackles')

# COUNT TACKLES Won BY PLAYER
tck_won = tackles.loc[[idx for idx, row in tackles.iterrows() if 703 in row['tags']]]
tck_won_by_player = tck_won.groupby('playerId').size().reset_index(name='n_tackles_won')

# SELECTION OF THE PLAYER IN THE FOURTH QUARTILE
best_tackler = tck_won_by_player[tck_won_by_player.n_tackles_won > tck_won_by_player.n_tackles_won.quantile(0.75)]

# RESULTS PREPARATION
results = best_tackler.merge(players, on=['playerId'])[['playerId', 'name', 'n_tackles_won']]
results.columns = COLUMNS_NAME
results.to_csv('problema_2_b.csv', index=False)
