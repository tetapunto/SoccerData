import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

input_folder = '../Input/'
output_folder = '../Output/'

lab_def_dict = dict(
    zip(range(5), ['Defensive full-back', 'Attacking full back', 'Ball playing defender', 'Sweeper/libero',
                   'Stopper']))
lab_fwd_dict = dict(zip(range(9), ['Cultured defensive midfielder', 'Ball winning midfielder', 'Defensive playmaker',
                                   'Box to box midfielder',
                                   'Inside midfielder', 'Wide midfielder', 'Runner', 'Playmaker',
                                   'Flamboyant attacking midfielder']))
lab_mid_dict = dict(
    zip(range(5), ['Explosive winger', 'Skillful winger', 'Complete forward', 'Cynical striker', 'Target man']))

if __name__ == '__main__':

    pca_def = joblib.load(output_folder + 'pca_df.pkl')
    km_def = joblib.load(output_folder + 'kmeans_df.pkl')
    pca_fwd = joblib.load(output_folder + 'pca_fw.pkl')
    km_fwd = joblib.load(output_folder + 'kmeans_fw.pkl')
    pca_mid = joblib.load(output_folder + 'pca_mf.pkl')
    km_mid = joblib.load(output_folder + 'kmeans_mf.pkl')
    pca_dict = dict(zip(['def', 'mid', 'fwd'], [pca_def, pca_mid, pca_fwd]))
    km_dict = dict(zip(['def', 'mid', 'fwd'], [km_def, km_mid, km_fwd]))

    df = pd.read_csv(input_folder + 'dataset.csv', index_col=0)
    players = pd.read_json(input_folder + 'players.json')
    teams = pd.read_json(input_folder + 'teams.json')

    # MERGE WITH PLAYERS TO GET THE CLASSIC ROLE
    players.role = [i['code2'] for i in players.role]
    df = df.merge(players[['role', 'wyId']], left_on='playerId', right_on='wyId', how='outer')

    # FILL NAN VALUES
    df.angle_mean.fillna(df.angle_mean.mean(), inplace=True)
    df.angle_std.fillna(0, inplace=True)
    df.height.fillna(df.height.mean(), inplace=True)
    df.weight.fillna(df.weight.mean(), inplace=True)
    df.ambidestro.fillna(df.ambidestro.mean(), inplace=True)
    df = df.fillna(0)

    # DROP UNUSEFUL COLUMNS
    df = df.drop(['matchPeriod', 'wyId'], axis=1, errors='ignore')

    # DROPPING OUT SOME NOISE COMING FROM POORLY VISITED CELLS, ENCODED IN CORRESPONDING
    col = ['00', '01', '02', '03',
           '04', '10', '11', '12', '13', '14', '20', '21', '22', '23', '24', '30',
           '31', '32', '33', '34', '40', '41', '42', '43', '44']


    def dropout(row):
        ix = row.values.argsort()[-3:][::-1]
        row.iloc[[i for i in range(len(row)) if i not in ix]] = 0
        return row

    def scaling(df):
        # SCALING COLUMNS WITH MAX VALUES HIGHER THAN 1
        scaler = MinMaxScaler()
        cols = ['standard_deviation', 'posizione_media_x', 'posizione_media_y',
                'angle_mean', 'angle_std', 'height', 'weight', 'yellow_card',
                'tempo_medio_tra_eventi']
        df_sc = scaler.fit_transform(df[cols])

        return pd.concat((
            pd.DataFrame(df_sc, columns=cols, index=df.index),
            df.drop(['posizione_media_x', 'posizione_media_y', 'standard_deviation',
                     'angle_mean', 'angle_std', 'height', 'weight', 'yellow_card',
                     'tempo_medio_tra_eventi'], axis=1)
        ), axis=1)


    res = [dropout(i[col]) for n, i in df.drop('playerId', 1).iterrows()]
    df[col] = pd.DataFrame(res, index=df.index)

    df_def = df.loc[df.role == 'DF'].copy().drop(['role', 'matchId'], 1)
    df_fwd = df.loc[df.role == 'FW'].copy().drop(['role', 'matchId'], 1)
    df_mid = df.loc[df.role == 'MD'].copy().drop(['role', 'matchId'], 1)
    df_other = df.loc[df.role.isnull()].copy().drop('role', 1)

    # THE PCA AND KM CLUSTERER DEPEND ON THE KWNOLEDGE OF THE CLASSICAL ROLE, SO WE HAVE TO SPLIT THE DATASET
    res = []
    if len(df_def):
        df_def = scaling(df_def)
        X = pca_def.transform(df_def.drop('playerId', axis=1))
        res.append(
            pd.Series(km_def.predict(X), name='label', index=df_def.index).map(lab_def_dict)
        )
    if len(df_fwd):
        df_fwd = scaling(df_fwd)
        X = pca_fwd.transform(df_fwd.drop('playerId', axis=1))
        res.append(
            pd.Series(km_fwd.predict(X), name='label', index=df_fwd.index).map(lab_fwd_dict)
        )
    if len(df_mid):
        df_mid = scaling(df_mid)
        X = pca_mid.transform(df_mid.drop('playerId', axis=1))
        res.append(
            pd.Series(km_mid.predict(X), name='label', index=df_mid.index).map(lab_mid_dict)
        )
    if len(df_other):
        df_other = scaling(df_other)
        scores = np.empty(3)
        for i, s in enumerate(['def', 'mid', 'fwd']):
            X = pca_dict[s].transform(df_mid.drop('playerId', axis=1))
            scores[i] = km_dict[s].score(X)
        role = ['def', 'mid', 'fwd'][scores.argmax()]
        dic = [lab_def_dict, lab_mid_dict, lab_fwd_dict][scores.argmax()]
        res.append(
            pd.Series(km_dict[role].predict(X), name='label', index=df_other.index).map(dic)
        )
    res = pd.concat(tuple(res))
    fin = df.groupby('playerId', as_index=False).agg({'matchId': 'nunique'}).rename(
        columns={'matchId': 'numero_presenze'}).merge(
        players[['wyId', 'shortName', 'currentTeamId']], left_on='playerId', right_on='wyId', how='left').merge(
        teams[['wyId', 'officialName']], left_on='currentTeamId', right_on='wyId', how='left')
    fin['wyrole'] = res
    fin[['playerId', 'shortName', 'officialName', 'wyrole', 'numero_presenze']] \
        .rename(columns={'shortName': 'nome_calciatore', 'officialName': 'nome_squadra', 'playerId': 'id_calciatore'}) \
        .sort_values('id_calciatore').drop_duplicates(['id_calciatore', 'wyrole']).to_csv(output_folder + 'sfida_1.csv',
                                                                                          index=False)
