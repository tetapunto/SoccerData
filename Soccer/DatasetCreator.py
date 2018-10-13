from functools import reduce

import numpy as np
import pandas as pd

# input_folder = '~/SoccerData/Input/'
input_folder = 'C:/Users/Davide/workspace/SoccerData/Input/'
output_folder = 'C:/Users/Davide/workspace/SoccerData/Output/'

list_measured_pct = [{'event_name': 'foul', 'Id': 2, 'subId': 20, 'tag': None, 'type': 'team'},
                     {'event_name': 'accurate_pass', 'Id': 8, 'subId': None, 'tag': 1801, 'type': 'player'},
                     {'event_name': 'accurate_shot', 'Id': 10, 'subId': 100, 'tag': 1801, 'type': 'player'},
                     {'event_name': 'duel', 'Id': 1, 'subId': None, 'tag': None, 'type': 'team'},
                     {'event_name': 'fk_corner', 'Id': 3, 'subId': 30, 'tag': None, 'type': 'team'},
                     {'event_name': 'fk_cross', 'Id': 3, 'subId': 32, 'tag': None, 'type': 'team'},
                     {'event_name': 'fk_shot', 'Id': 3, 'subId': 33, 'tag': None, 'type': 'team'},
                     {'event_name': 'foul', 'Id': 2, 'subId': 20, 'tag': None, 'type': 'team'},
                     {'event_name': 'interception', 'Id': None, 'subId': None, 'tag': 1401, 'type': 'team'},
                     {'event_name': 'key_pass', 'Id': None, 'subId': None, 'tag': 302, 'type': 'team'},
                     {'event_name': 'offside', 'Id': 6, 'subId': 60, 'tag': None, 'type': 'team'},
                     {'event_name': 'pass_cross', 'Id': 8, 'subId': 80, 'tag': None, 'type': 'team'},
                     {'event_name': 'pass_high', 'Id': 8, 'subId': 83, 'tag': None, 'type': 'team'},
                     {'event_name': 'pass_simple', 'Id': 8, 'subId': 85, 'tag': None, 'type': 'team'},
                     {'event_name': 'pass_smart', 'Id': 8, 'subId': 86, 'tag': None, 'type': 'team'},
                     {'event_name': 'shot', 'Id': 10, 'subId': 100, 'tag': None, 'type': 'team'}]

list_annual_event = [{'event_name': 'assist', 'Id': None, 'subId': None, 'tag': 301},
                     {'event_name': 'gol', 'Id': None, 'subId': None, 'tag': 101},
                     {'event_name': 'yellow_card', 'Id': None, 'subId': None, 'tag': 1702}]


def make_list(list_dict):
    """Extrapolated tag Id from list of dictionaries"""
    if len(list_dict) == 0:
        return []
    return [i['id'] for i in list_dict]


def select_from_event(df, Id=None, subId=None, tag=None):
    """Given one between Id, subId or tag extract sub dataframe from dataset events"""
    tag_list = [tag in i for i in df.tags] if tag is not None else [not tag in i for i in df.tags]
    id_list = Id if Id is not None else df.eventId
    subid_list = subId if subId is not None else df.subEventId
    return df[(df.eventId == id_list) & tag_list & (df.subEventId == subid_list)]


# df raggruppato per partita + finestra temporale
def get_percentage_event_wrt(df, Id=None, subId=None, tag=None, event_name='name'):
    """Return the percentage of a specific event made by a player with respect to all team events of the same kind
     in a given match, time window"""
    df_by_event = select_from_event(df, Id, subId, tag)
    counter = df_by_event.groupby(['playerId', 'teamId']).size().reset_index().merge(
        df_by_event.groupby('teamId').size().reset_index(), on='teamId')
    counter.rename(columns={'0_x': 'count_by_player', '0_y': 'total'}, inplace=True)
    counter[event_name + '_pct_WRT'] = counter.count_by_player / counter.total
    return counter.drop(['count_by_player', 'total', 'teamId'], 1)


# df raggruppato per partita + finestra temporale
def get_percentage_event_wrp(df, Id=None, subId=None, tag=None, event_name='name'):
    """Return the percentage of a specific event made by a player with respect to all his events
     in a given match, time window"""
    df_by_event = select_from_event(df, Id, subId, tag)
    counter = df_by_event.groupby(['playerId', 'teamId']).size().reset_index().merge(
        df.groupby(['playerId', 'teamId']).size().reset_index(), on=['teamId', 'playerId'])
    counter.rename(columns={'0_x': 'single_ev_player', '0_y': 'total_ev_player'}, inplace=True)
    counter[event_name + '_pct_WRP'] = counter.single_ev_player / counter.total_ev_player
    return counter.drop(['single_ev_player', 'total_ev_player', 'teamId'], 1)


def annual_mean(total_events, Id=None, subId=None, tag=None, event_name=''):
    """Return the annual mean of a given event for each player"""
    df = select_from_event(total_events, Id, subId, tag)
    res = (df.groupby('playerId').size() / total_events.groupby(['playerId']).matchId.nunique()).reset_index().rename(
        columns={0: event_name})
    return res


def get_pct(df, index, Id=None, subId=None, tag=None, event_name='name', type='team'):
    if type == 'team':
        r1 = get_percentage_event_wrt(df, Id, subId, tag, event_name)
    else:
        r1 = get_percentage_event_wrp(df, Id, subId, tag, event_name)
    r1['matchId'] = index[0]
    r1['matchPeriod'] = index[1]
    return r1


def create_single_percentage(event, Id, subId, tag, event_name=None, type=''):
    res = []
    print(event_name)
    for ix, df in event.groupby(['matchId', 'matchPeriod']):
        res.append(get_pct(df, ix, Id=Id, subId=subId, tag=tag, event_name=event_name, type=type))
    return pd.concat(res)


def mean_and_dispersion_pos(series_pos):
    """This function receives a Series of list of dicts and computes the x mean, the y mean and the
        mean squared distance. Returns a pandas DataFrame"""
    ys = np.array([line[0]['y'] for line in series_pos.values])  # for d in line
    xs = np.array([line[0]['x'] for line in series_pos.values])  # for d in line
    y_mean = ys.mean()
    x_mean = xs.mean()
    dist = np.sqrt(np.power(ys - y_mean, 2) + np.power(xs - x_mean, 2))
    return pd.DataFrame([[x_mean, y_mean, dist.mean()]],
                        columns=['posizione_media_x', 'posizione_media_y', 'standard_deviation'])


def extract_time_delta(series_time):
    """Computes difference between subsequent events"""
    series_time = series_time.sort_values()
    return np.sum(series_time - series_time.shift(1))


def angle_check(pos_tup):
    y_end = pos_tup[0]
    y_start = pos_tup[1]
    x_end = pos_tup[2]
    x_start = pos_tup[3]

    deltay = y_end - y_start
    deltax = x_end - x_start
    if deltay >= 0:
        if deltax == 0.:
            return np.pi / 2
        elif deltax > 0:
            return np.arctan(deltay / deltax)
        else:
            return np.pi - abs(np.arctan(deltay / deltax))
    else:
        if deltax == 0:
            return 3 * np.pi / 2
        elif deltax > 0:
            return 2 * np.pi - abs(np.arctan(deltay / deltax))
        else:
            return np.pi + abs(np.arctan(deltay / deltax))


def distance(d1, d2):
    return np.sqrt((d1['x'] - d2['x']) ** 2 + (d1['y'] - d2['y']) ** 2)


def measure_total_distance(df):
    res = []
    ravel = df.positions.values.ravel()
    for i in range(len(ravel) - 1):
        res.append(distance(ravel[i][0], ravel[i + 1][0]))
    return sum(res)


def event_by_cell(df, ix):
    tt = df.groupby(['numero_cella_x', 'numero_cella_y']).size().reset_index()
    tt.index = [str(i) + str(j) for i, j in zip(tt.numero_cella_x, tt.numero_cella_y)]
    tt = tt[0]
    tt['matchId'] = ix[0]
    tt['matchPeriod'] = ix[1]
    tt['playerId'] = ix[2]
    return tt


def get_x(lista):
    if not lista:
        return None
    return lista[0]['x']


def get_y(lista):
    if not lista:
        return None
    return lista[0]['y']


if __name__ == '__main__':
    print('Importing')

    player = pd.read_json(input_folder + 'players.json')
    event = pd.read_json(input_folder + 'events.json')
    event.tags = event.tags.apply(make_list)
    extrapolate_pos = lambda l: [v for d in l for v in d.values()]
    event[['y_start', 'x_start', 'y_end', 'x_end']] = pd.DataFrame(event.positions.apply(extrapolate_pos).tolist(),
                                                                   index=event.index)
    match = pd.read_json(input_folder + 'matches.json')
    team = pd.read_json(input_folder + 'teams.json')

    print('Evaluating Percentage')
    res_pct = [create_single_percentage(event, **d) for d in list_measured_pct]
    print('Merging percentages')
    df1 = reduce(lambda x, y: pd.merge(x, y, on=['matchId', 'playerId', 'matchPeriod'], how='outer'), res_pct)

    print('Evaluating Annual Measures')
    res_annual = [annual_mean(event, **d) for d in list_annual_event]
    df2 = reduce(lambda x, y: pd.merge(x, y, on='playerId', how='outer'), res_annual)
    print('writing')
    df1 = df1.merge(df2, on='playerId', how='outer')
    print(df1.shape)

    print('Evaluating Avg position and dispersion')
    result1 = event.groupby(['matchId', 'playerId', 'matchPeriod']).positions.apply(mean_and_dispersion_pos)
    result1.index = result1.index.droplevel(3)
    result1 = result1.reset_index().merge(player[['wyId', 'height', 'weight']], left_on=['playerId'], right_on=['wyId'],
                                          how='left').drop('wyId', axis=1)
    df1 = df1.merge(result1, on=['matchId', 'playerId', 'matchPeriod'], how='outer')
    print(df1.shape)

    print('Evaluating mean event rate')
    result2 = event.groupby(['matchId', 'playerId', 'matchPeriod'], as_index=False).agg(
        {'eventSec': extract_time_delta}).merge(
        event.groupby(['matchId', 'playerId', 'matchPeriod'], as_index=False).agg({'id': 'nunique'}),
        on=['matchId', 'playerId', 'matchPeriod']).set_index(['matchId', 'playerId', 'matchPeriod']).assign(
        tempo_medio_tra_eventi=lambda x: x.eventSec / x.id).loc[:, ['tempo_medio_tra_eventi']]
    df1 = df1.merge(result2.reset_index(), on=['matchId', 'playerId', 'matchPeriod'], how='outer')
    print(df1.shape)

    print('Evaluating level of ambidestrism')
    result3 = select_from_event(event, tag=401).groupby(['playerId'], as_index=False).agg({'id': np.size}).rename(
        columns={'id': 'left'}).merge(
        select_from_event(event, tag=402).groupby(['playerId'], as_index=False).agg({'id': np.size}).rename(
            columns={'id': 'right'}), on=['playerId'], how='outer').fillna(0).assign(
        ambidestro=lambda x: (x.left - x.right).abs() / (x.left + x.right)).iloc[:, [0, 3]]
    df1 = df1.merge(result3, on='playerId', how='outer')
    print(df1.shape)

    print('Evaluating passages angles mean')
    event['has_doublepos'] = event.positions.apply(lambda l: True if len(l) >= 2 else False)
    event.groupby('eventName').has_doublepos.sum() * 100 / event.groupby('eventName').id.size()
    sub = event[event.has_doublepos].copy()
    event.loc[event.has_doublepos, 'angle'] = event.loc[event.has_doublepos, ['y_end',
                                                                              'y_start',
                                                                              'x_end',
                                                                              'x_start']].apply(angle_check, axis=1)
    angdf = event.groupby(['matchId', 'playerId', 'matchPeriod'], as_index=False).agg({'angle': [np.mean, np.std]})
    angdf.columns = ['matchId', 'playerId', 'matchPeriod', 'angle_mean', 'angle_std']
    df1 = df1.merge(angdf, on=['matchId', 'playerId', 'matchPeriod'], how='outer')
    print(df1.shape)

    print('Evaluating total distance')
    event.positions = [i if len(i) == 2 else [i[0], i[0]] for i in event.positions.values]
    dist = event.groupby(['matchId', 'matchPeriod', 'playerId']).apply(measure_total_distance).reset_index().rename(
        columns={0: 'total_distance'})
    df1 = df1.merge(dist, on=['matchId', 'playerId', 'matchPeriod'], how='outer')
    print(df1.shape)

    print('Evaluating events by grid')
    # FROM POSITIONS TO SINGLE VALUE COLUMNS x,y
    event['x'] = event.positions.apply(get_x)
    event['y'] = event.positions.apply(get_y)
    event.drop('positions', axis=1, inplace=True)

    # FROM FIELD POSITIONS TO GRID
    bins = np.arange(0, 120, 20)
    label = [0, 1, 2, 3, 4]
    event['numero_cella_x'] = pd.cut(event.x, bins=bins, labels=label).fillna(0.).astype(int)
    event['numero_cella_y'] = pd.cut(event.y, bins=bins, labels=label).fillna(0.).astype(int)

    res = []
    for ix, i in event.groupby(['matchId', 'matchPeriod', 'playerId']):
        res.append(event_by_cell(i, ix))
    r = pd.concat(res, 1).T.fillna(0)
    df1 = df1.merge(r, on=['matchId', 'playerId', 'matchPeriod'], how='outer')
    print(df1.shape)

    df1.to_csv(output_folder + 'dataset.csv')
