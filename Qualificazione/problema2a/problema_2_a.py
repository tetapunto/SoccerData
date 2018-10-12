import pandas as pd

COLUMNS_NAME = ['nome_squadra', 'accuratezza_media_dei_passaggi', 'standard_deviation_accuratezza_media_dei_passaggi']


def make_list(list_dict, key='id'):
    if len(list_dict) == 0:
        return []
    return [i[key] for i in list_dict]


# IMPORT DATA
data = pd.read_json('worldCup.json')
players = pd.read_json('player.json')

# CLEAN TAGS
data.tags = data.tags.apply(make_list)

# TOTAL PASS BY PLAYER
pass_by_player = data[data.eventId == 8].groupby(['playerId', 'teamId']).size().reset_index(name='total_pass')

# ALL ACCURATE EVENTS
accurate_events = data.loc[[idx for idx, row in data.iterrows() if 1801 in row['tags']]]

# ACCURATE PASS
accurate_pass_by_player = accurate_events[accurate_events.eventId == 8].groupby(
    ['playerId', 'teamId']).size().reset_index(name='accurate_pass')

# STATISTICS
pass_summary = pass_by_player.merge(accurate_pass_by_player, on=['playerId', 'teamId'])
pass_summary['accuracy'] = pass_summary.accurate_pass / pass_summary.total_pass
mean_accuray_by_team = pass_summary.groupby('teamId')['accuracy'].mean().reset_index(name='mean_accuracy_by_team')
std_accuracy_by_team = pass_summary.groupby('teamId')['accuracy'].std().reset_index(name='std_accuracy_by_team')

# RESULTS EXPORT PREPARATION
result = mean_accuray_by_team.merge(std_accuracy_by_team, on=['teamId'])
result.columns = COLUMNS_NAME
result = result.round({'accuratezza_media_dei_passaggi': 2, 'standard_deviation_accuratezza_media_dei_passaggi': 2})
result.to_csv('problema_2_a.csv', index=False)
