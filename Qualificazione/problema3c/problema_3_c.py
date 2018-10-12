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

    df_FR = df[df.nome_squadra == 'France']
    df_CR = df[df.nome_squadra == 'Croatia']
    all = df_CR.merge(df_FR, on=['numero_cella_x', 'numero_cella_y'], suffixes=('_cr', '_fr'))

    # TOTAL EVENTS
    ev = all[['numero_cella_x', 'numero_cella_y', 'frequenza_eventi_cr', 'frequenza_eventi_fr']].copy()
    ev['differenza_di_frequenza'] = np.abs(ev.frequenza_eventi_cr - ev.frequenza_eventi_fr)
    ev['squadra_dominante'] = ['Croatia' if (i - j) > 0 else np.nan if (i - j) == 0 else 'France' for i, j in
                               zip(ev.frequenza_eventi_cr, ev.frequenza_eventi_fr)]
    ev.drop(['frequenza_eventi_cr', 'frequenza_eventi_fr'], 1, inplace=True)
    ev['tipo_griglia'] = [1 for i in range(len(ev))]

    # PASSAGES
    pas = all[
        ['numero_cella_x', 'numero_cella_y', 'frequenza_passaggi_accurati_cr', 'frequenza_passaggi_accurati_fr']].copy()
    pas['differenza_di_frequenza'] = np.abs(pas.frequenza_passaggi_accurati_cr - pas.frequenza_passaggi_accurati_fr)
    pas['squadra_dominante'] = ['Croatia' if (i - j) > 0 else np.nan if (i - j) == 0 else 'France' for i, j in
                                zip(pas.frequenza_passaggi_accurati_cr, pas.frequenza_passaggi_accurati_fr)]
    pas.drop(['frequenza_passaggi_accurati_cr', 'frequenza_passaggi_accurati_fr'], 1, inplace=True)
    pas['tipo_griglia'] = [2 for i in range(len(pas))]

    # SHOTS
    shots = all[['numero_cella_x', 'numero_cella_y', 'frequenza_tiri_cr', 'frequenza_tiri_fr']].copy()
    shots['differenza_di_frequenza'] = np.abs(shots.frequenza_tiri_cr - shots.frequenza_tiri_fr)
    shots['squadra_dominante'] = ['Croatia' if (i - j) > 0 else np.nan if (i - j) == 0 else 'France' for i, j in
                                  zip(shots.frequenza_tiri_cr, shots.frequenza_tiri_fr)]
    shots.drop(['frequenza_tiri_cr', 'frequenza_tiri_fr'], 1, inplace=True)
    shots['tipo_griglia'] = [3 for i in range(len(shots))]

    # FOULS
    fouls = all[['numero_cella_x', 'numero_cella_y', 'frequenza_falli_subiti_cr', 'frequenza_falli_subiti_fr']].copy()
    fouls['differenza_di_frequenza'] = np.abs(fouls.frequenza_falli_subiti_cr - fouls.frequenza_falli_subiti_fr)
    fouls['squadra_dominante'] = ['Croatia' if (i - j) > 0 else np.nan if (i - j) == 0 else 'France' for i, j in
                                  zip(fouls.frequenza_falli_subiti_cr, fouls.frequenza_falli_subiti_fr)]
    fouls.drop(['frequenza_falli_subiti_cr', 'frequenza_falli_subiti_fr'], 1, inplace=True)
    fouls['tipo_griglia'] = [4 for i in range(len(fouls))]

    result = pd.concat([ev, pas, shots, fouls])
    result.to_csv('problema_3_c.csv', index=False)
