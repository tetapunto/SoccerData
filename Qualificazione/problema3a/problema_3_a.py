import pandas as pd
import numpy as np

COLUMNS_NAME = ['nome_squadra', 'numero_cella_x', 'numero_cella_y', 'frequenza_eventi', 'frequenza_passaggi_accurati',
                'frequenza_tiri', 'frequenza_falli_subiti']


# HELP FUNCTION
def make_list(list_dict, key='id'):
    if len(list_dict) == 0:
        return []
    return [i[key] for i in list_dict]


def get_x(lista):
    if not lista:
        return None
    return lista[0]['x']


def get_y(lista):
    if not lista:
        return None
    return lista[0]['y']


def get_frames():
    # IMPORT DATA
    data = pd.read_json('worldCup.json')
    players = pd.read_json('player.json')
    data = data[data.playerId != 0]

    data.tags = data.tags.apply(make_list)

    # FROM POSITIONS TO SINGLE VALUE COLUMNS x,y
    data['x'] = data.positions.apply(get_x)
    data['y'] = data.positions.apply(get_y)
    data.drop('positions', axis=1, inplace=True)

    # FROM FIELD POSITIONS TO GRID
    bins = np.arange(0, 120, 20)
    label = [0, 1, 2, 3, 4]
    data['numero_cella_x'] = pd.cut(data.x, bins=bins, labels=label).fillna(0.).astype(int)
    data['numero_cella_y'] = pd.cut(data.y, bins=bins, labels=label).fillna(0.).astype(int)

    # EVENTI TOTALI
    total_events_by_team = data.groupby(['teamId']).size().reset_index(name='total_events_cell')
    total_events_by_cell = data.groupby(['numero_cella_x', 'numero_cella_y', 'teamId']).size().reset_index(
        name='number_events_cell')

    df = total_events_by_cell.merge(total_events_by_team, on=['teamId'])
    df['events_freq'] = df.number_events_cell / df.total_events_cell
    df.drop(['number_events_cell', 'total_events_cell'], 1, inplace=True)

    # PASSAGGI
    accurate_pass = data.loc[[idx for idx, row in data.iterrows() if 1801 in row['tags'] and row.eventId == 8]]

    accurate_pass_by_team = accurate_pass.groupby(['teamId']).size().reset_index(name='num_total_pass')
    accurate_pass_by_cell = accurate_pass.groupby(['numero_cella_x', 'numero_cella_y', 'teamId']).size().reset_index(
        name='number_acc_pass')

    df_pass = accurate_pass_by_cell.merge(accurate_pass_by_team, on=['teamId'])
    df_pass['acc_pass_freq'] = df_pass.number_acc_pass / df_pass.num_total_pass

    df = df.merge(df_pass[['numero_cella_x', 'numero_cella_y', 'teamId', 'acc_pass_freq']], how='left',
                  on=['numero_cella_x', 'numero_cella_y', 'teamId'])

    # TIRI
    shot = data.loc[data.eventId == 10]

    shot_by_team = shot.groupby(['teamId']).size().reset_index(name='num_total_shot')
    shot_by_cell = shot.groupby(['numero_cella_x', 'numero_cella_y', 'teamId']).size().reset_index(name='number_shot')

    df_shot = shot_by_cell.merge(shot_by_team, on=['teamId'])
    df_shot['shot_freq'] = df_shot.number_shot / df_shot.num_total_shot

    df = df.merge(df_shot[['numero_cella_x', 'numero_cella_y', 'teamId', 'shot_freq']], how='left',
                  on=['numero_cella_x', 'numero_cella_y', 'teamId'])

    # FALLI PER PASSARE DA FALLI EFFETTUATI A FALLI SUBITI INVERTIAMO GRIGLIE CAMPO E NOMI SQUADRE
    foul = data.loc[(data.eventId == 2) & (data.subEventId == 20)]

    foul_by_team = foul.groupby(['teamId']).size().reset_index(name='num_total_foul')
    foul_by_team['teamId'] = ['France' if i == 'Croatia' else 'Croatia' for i in foul_by_team.teamId]

    foul_by_cell_start = foul.groupby(['numero_cella_x', 'numero_cella_y', 'teamId']).size().reset_index(
        name='number_foul')

    foul_by_cell = foul_by_cell_start.copy()
    foul_by_cell['numero_cella_x'] = np.abs(foul_by_cell['numero_cella_x'].astype(int) - 4)
    foul_by_cell['numero_cella_y'] = np.abs(foul_by_cell['numero_cella_y'].astype(int) - 4)
    foul_by_cell['teamId'] = ['France' if i == 'Croatia' else 'Croatia' for i in foul_by_cell.teamId]

    df_foul = foul_by_cell.merge(foul_by_team, on=['teamId'])
    df_foul['foul_freq'] = df_foul.number_foul / df_foul.num_total_foul

    df = df.merge(df_foul[['numero_cella_x', 'numero_cella_y', 'teamId', 'foul_freq']], how='left',
                  on=['numero_cella_x', 'numero_cella_y', 'teamId'])

    # RESULTS PREPARATION
    df = df[['teamId', 'numero_cella_x', 'numero_cella_y', 'events_freq', 'acc_pass_freq', 'shot_freq', 'foul_freq']]
    df = df.round({i: 2 for i in COLUMNS_NAME[3:]})
    df.columns = COLUMNS_NAME
    return df.fillna(0)


if __name__ == '__main__':
    df = get_frames()
    df.to_csv('problema_3_a.csv', index=False)
