"""Job d'entraînement FPL : à lancer UNE FOIS par gameweek (10 à 20 min).

    python train_job.py

Reprend les sections 1 à 10 et 15 du notebook, sans rien de personnel (pas de TEAM_ID).
Sortie : data/latest.pkl (lu par l'API) et data/gw{N}.pkl (archive).
"""
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
import requests
from xgboost import XGBRegressor
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from app.tips import compute_tips_df

CURRENT_SEASON = '2026-27'
PAST_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
API = 'https://fantasy.premierleague.com/api'
BASE = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data'
OUT_DIR = Path(__file__).parent / 'data'
W_COMP = 0.5            # mélange modèle "par composantes" / modèle direct (0.5 = moitié-moitié)

KEEP = ['fixture','round','kickoff_time','was_home','opponent_team','team','name','position',
        'minutes','total_points','goals_scored','assists','clean_sheets','goals_conceded','saves',
        'bonus','bps','influence','creativity','threat','ict_index',
        'expected_goals','expected_assists','expected_goal_involvements','expected_goals_conceded',
        'value','team_h_score','team_a_score']
ROLL = ['total_points','minutes','full90','expected_goals','expected_assists','expected_goal_involvements',
        'expected_goals_conceded','bps','bonus','threat','creativity','influence','ict_index',
        'goals_scored','assists','clean_sheets','saves']
FEATURES = [f'{c}_r{w}' for w in (3,5,10) for c in ROLL] + \
           ['games_seen','rest_days','pos_code','home','value','own_gf6','own_ga6','opp_gf6','opp_ga6']
NEW = ['own_xg6','own_xga6','own_xg12','own_xga12','opp_xg6','opp_xga6','opp_xg12','opp_xga12',
       'xgi90_r5','xgi90_r10','xg90_r5','xg90_r10','bps90_r5','bps90_r10',
       'pts_per_start_r10','xgi_vs_opp','cs_odds_proxy','att_edge']
FEATURES2 = FEATURES + NEW
TXCOLS = ['xg6','xga6','xg12','xga12']
GOAL_PTS = {'GKP':10, 'DEF':6, 'MID':5, 'FWD':4}
CS_PTS = {'GKP':4, 'DEF':4, 'MID':1, 'FWD':0}
COMPS = ['t_app','t_goals','t_assists','t_cs','t_gc','t_saves','t_bonus','t_other']
P_XGB = dict(learning_rate=0.04, max_depth=5, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=10, reg_lambda=2.0, n_jobs=-1, random_state=42)


# ---------------------------------------------------------------- 1-2. Données
def load_data():
    past = []
    for s in PAST_SEASONS:
        try:
            gw = pd.read_csv(f'{BASE}/{s}/gws/merged_gw.csv', on_bad_lines='skip', low_memory=False)
            teams = pd.read_csv(f'{BASE}/{s}/teams.csv')
            gw['opponent_team'] = gw['opponent_team'].map(dict(zip(teams['id'], teams['name'])))
            gw['season'] = s
            past.append(gw[['season'] + KEEP])
            print(s, len(gw), 'lignes OK')
        except Exception as e:
            print(s, 'échec :', e)
    past = pd.concat(past, ignore_index=True)

    boot = requests.get(f'{API}/bootstrap-static/', timeout=30).json()
    players = pd.DataFrame(boot['elements'])
    teams_live = pd.DataFrame(boot['teams'])
    events = pd.DataFrame(boot['events'])
    fixtures = pd.DataFrame(requests.get(f'{API}/fixtures/', timeout=30).json())

    team_name = dict(zip(teams_live['id'], teams_live['name']))
    players['name'] = players['first_name'] + ' ' + players['second_name']
    players['position'] = players['element_type'].map({1:'GKP', 2:'DEF', 3:'MID', 4:'FWD'})
    players['team_name'] = players['team'].map(team_name)
    next_gw = int(events.loc[events['is_next'], 'id'].iloc[0])
    print('Prochaine GW :', next_gw)

    def get_hist(pid):
        for _ in range(4):
            try:
                r = requests.get(f'{API}/element-summary/{pid}/', timeout=20)
                if r.status_code == 200:
                    return pd.DataFrame(r.json()['history'])
            except Exception:
                pass
            time.sleep(1)
        return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=8) as ex:
        hists = list(ex.map(get_hist, players['id']))

    # Pour l'API : points par (joueur, GW) et prix de départ (calcul du prix de vente)
    pts_gw, first_value = {}, {}
    for hh in hists:
        if len(hh):
            e = int(hh['element'].iloc[0])
            for rd, v in hh.groupby('round')['total_points'].sum().items():
                pts_gw[(e, int(rd))] = int(v)
            r1 = hh.loc[hh['round'] == 1, 'value']
            if len(r1):
                first_value[e] = int(r1.iloc[0])

    live = pd.concat([hh for hh in hists if len(hh)], ignore_index=True)
    live = live.drop(columns=['name','position','team'], errors='ignore') \
               .join(players.set_index('id')[['name','position','team_name']], on='element')
    live = live.rename(columns={'team_name': 'team'})
    live['opponent_team'] = live['opponent_team'].map(team_name)
    live['season'] = CURRENT_SEASON
    live = live[['season'] + KEEP]

    hist = pd.concat([past, live], ignore_index=True)
    num = [c for c in KEEP if c not in ('fixture','round','kickoff_time','was_home','opponent_team','team','name','position')]
    for c in num:
        hist[c] = pd.to_numeric(hist[c], errors='coerce')
    hist['position'] = hist['position'].replace({'GK': 'GKP'})
    hist = hist[hist['position'].isin(['GKP','DEF','MID','FWD'])]
    hist['kickoff_time'] = pd.to_datetime(hist['kickoff_time'], utc=True, errors='coerce').astype('datetime64[ns, UTC]')
    hist['was_home'] = hist['was_home'].astype(str).str.lower().eq('true')
    hist = hist.dropna(subset=['kickoff_time', 'opponent_team'])
    hist['pkey'] = hist['name']
    hist['is_future'] = False
    return dict(players=players, events=events, fixtures=fixtures, team_name=team_name,
                next_gw=next_gw, hist=hist, pts_gw=pts_gw, first_value=first_value)


# ---------------------------------------------------------------- 3. Features v1
def add_team_form(frame, tm, team_col, prefix):
    state = tm[['team','kickoff_time','gf6','ga6']].rename(
        columns={'team': team_col, 'kickoff_time': 'state_time',
                 'gf6': f'{prefix}_gf6', 'ga6': f'{prefix}_ga6'}).sort_values('state_time')
    frame = frame.sort_values('kickoff_time')
    return pd.merge_asof(frame, state, left_on='kickoff_time', right_on='state_time',
                         by=team_col, allow_exact_matches=False).drop(columns='state_time')


def build_features(hist):
    tm = hist.copy()
    tm['gf'] = np.where(tm['was_home'], tm['team_h_score'], tm['team_a_score'])
    tm['ga'] = np.where(tm['was_home'], tm['team_a_score'], tm['team_h_score'])
    tm = (tm.drop_duplicates(['season','fixture','team'])[['team','kickoff_time','gf','ga']]
            .dropna().sort_values(['team','kickoff_time']))
    tm['gf6'] = tm.groupby('team')['gf'].transform(lambda s: s.rolling(6, min_periods=2).mean())
    tm['ga6'] = tm.groupby('team')['ga'].transform(lambda s: s.rolling(6, min_periods=2).mean())

    df = add_team_form(hist, tm, 'team', 'own')
    df = add_team_form(df, tm, 'opponent_team', 'opp')
    df = df.sort_values(['pkey','kickoff_time','fixture']).reset_index(drop=True)

    df['full90'] = (df['minutes'] >= 60).astype(float).where(df['minutes'].notna())
    shifted = df.groupby('pkey')[ROLL].shift(1)       # shift(1) : jamais de fuite du futur
    blocks = []
    for w in (3, 5, 10):
        r = shifted.groupby(df['pkey']).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        r = r.reindex(df.index)
        r.columns = [f'{c}_r{w}' for c in ROLL]
        blocks.append(r)
    df = pd.concat([df] + blocks, axis=1)
    g = df.groupby('pkey')
    df['games_seen'] = g.cumcount()
    df['rest_days'] = g['kickoff_time'].diff().dt.days.clip(upper=30)
    df['pos_code'] = df['position'].map({'GKP':0, 'DEF':1, 'MID':2, 'FWD':3})
    df['home'] = df['was_home'].astype(int)
    return df, tm


# ---------------------------------------------------------------- 8. Features v2
def build_tx(hist):
    tx = (hist.groupby(['season','fixture','team'])
              .agg(kickoff_time=('kickoff_time','first'),
                   xg=('expected_goals', lambda s: s.sum(min_count=1)))
              .reset_index())
    opp_x = tx[['season','fixture','team','xg']].rename(columns={'team':'opp_team','xg':'xga'})
    tx = tx.merge(opp_x, on=['season','fixture'])
    tx = tx[tx['team'] != tx['opp_team']].dropna(subset=['xg','xga']).sort_values(['team','kickoff_time'])
    for w in (6, 12):
        tx[f'xg{w}'] = tx.groupby('team')['xg'].transform(lambda s: s.rolling(w, min_periods=3).mean())
        tx[f'xga{w}'] = tx.groupby('team')['xga'].transform(lambda s: s.rolling(w, min_periods=3).mean())
    return tx


def add_team_xg(frame, tx):
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith(('own_xg','opp_xg'))])
    f = frame.sort_values('kickoff_time')
    for col, prefix in (('team','own'), ('opponent_team','opp')):
        st = tx[['team','kickoff_time'] + TXCOLS].rename(
            columns={'team': col, 'kickoff_time': 'st_time', **{c: f'{prefix}_{c}' for c in TXCOLS}}
        ).sort_values('st_time')
        f = pd.merge_asof(f, st, left_on='kickoff_time', right_on='st_time',
                          by=col, allow_exact_matches=False).drop(columns='st_time')
        f = f.sort_values('kickoff_time')
    return f


def add_derived(f):
    f = f.copy()
    for w in (5, 10):
        mins = (f[f'minutes_r{w}'] / 90).where(f[f'minutes_r{w}'] >= 15)
        f[f'xgi90_r{w}'] = f[f'expected_goal_involvements_r{w}'] / mins
        f[f'xg90_r{w}'] = f[f'expected_goals_r{w}'] / mins
        f[f'bps90_r{w}'] = f[f'bps_r{w}'] / mins
    f['pts_per_start_r10'] = f['total_points_r10'] / f['full90_r10'].replace(0, np.nan)
    f['xgi_vs_opp'] = f['expected_goal_involvements_r5'] * f['opp_xga6']
    f['cs_odds_proxy'] = f['own_xga6'] - f['opp_xg6']
    f['att_edge'] = f['own_xg6'] - f['opp_xga6']
    return f


def add_targets(df2):
    pm = df2['position']
    p60 = df2['minutes'] >= 60
    df2['t_app'] = (df2['minutes'] > 0).astype(float) + p60.astype(float)
    df2['t_goals'] = df2['goals_scored'] * pm.map(GOAL_PTS)
    df2['t_assists'] = df2['assists'] * 3
    df2['t_cs'] = df2['clean_sheets'].clip(upper=1) * pm.map(CS_PTS) * p60
    df2['t_gc'] = -(df2['goals_conceded'] // 2) * pm.isin(['GKP','DEF'])
    df2['t_saves'] = (df2['saves'] // 3) * (pm == 'GKP')
    df2['t_bonus'] = df2['bonus']
    df2['t_other'] = df2['total_points'] - df2[['t_app','t_goals','t_assists','t_cs','t_gc','t_saves','t_bonus']].sum(axis=1)
    return df2


def recency(d, ref):      # demi-vie ~ 1 an : les matchs récents comptent plus
    age = (ref - d['kickoff_time']).dt.days.clip(lower=0)
    return 0.5 ** (age / 365)


def fit_pair(d, target, ref, n=300):
    X, y, wt = d[FEATURES2], d[target], recency(d, ref)
    models = [XGBRegressor(n_estimators=n, **P_XGB).fit(X, y, sample_weight=wt)]
    if HAS_LGB:
        m2 = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.04, num_leaves=31, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=2.0, random_state=42, verbose=-1)
        models.append(m2.fit(X, y, sample_weight=wt))
    return lambda f: np.mean([m.predict(f[FEATURES2]) for m in models], axis=0)


# ---------------------------------------------------------------- 6. Horizon + calendrier
def build_horizon_rows(D, hist, tm, tx):
    players, events, fixtures, team_name, next_gw = D['players'], D['events'], D['fixtures'], D['team_name'], D['next_gw']
    last_gw = 19 if next_gw <= 19 else 38
    horizon = list(range(next_gw, last_gw + 1))
    fxh = fixtures[fixtures['event'].isin(horizon)].copy()

    cal = pd.concat([fxh[['event','team_h']].rename(columns={'team_h':'team'}),
                     fxh[['event','team_a']].rename(columns={'team_a':'team'})])
    cnt = (cal.groupby(['event','team']).size().unstack(fill_value=0)
              .reindex(index=horizon, columns=list(team_name.keys()), fill_value=0))
    dl = events.set_index('id')['deadline_time']
    calendar = pd.DataFrame({
        'matchs': cnt.sum(axis=1),
        'equipes_qui_jouent': (cnt > 0).sum(axis=1),
        'equipes_DGW': [', '.join(team_name[t] for t in cnt.columns[cnt.loc[g] >= 2]) for g in horizon],
        'equipes_BGW': [', '.join(team_name[t] for t in cnt.columns[cnt.loc[g] == 0]) for g in horizon],
        'deadline': [dl[g] for g in horizon],
    })

    # État actuel de chaque joueur / équipe
    h = hist.sort_values(['pkey','kickoff_time']).copy()
    h['full90'] = (h['minutes'] >= 60).astype(float)
    state = pd.DataFrame(index=h['pkey'].unique())
    for w in (3, 5, 10):
        t = h.groupby('pkey').tail(w).groupby('pkey')[ROLL].mean()
        t.columns = [f'{c}_r{w}' for c in ROLL]
        state = state.join(t)
    state['games_seen'] = h.groupby('pkey').size()
    tstate = tm.sort_values('kickoff_time').groupby('team').tail(1).set_index('team')[['gf6','ga6']]

    home = fxh[['id','event','team_h','team_a']].rename(columns={'id':'fixture','team_h':'team_id','team_a':'opp_id'}); home['home'] = 1
    away = fxh[['id','event','team_a','team_h']].rename(columns={'id':'fixture','team_a':'team_id','team_h':'opp_id'}); away['home'] = 0
    pl2 = players[['id','name','position','team','now_cost','status','chance_of_playing_next_round']] \
            .rename(columns={'id':'element','team':'team_id'})
    hz = pd.concat([home, away]).merge(pl2, on='team_id').join(state, on='name')
    hz['team'] = hz['team_id'].map(team_name); hz['opp'] = hz['opp_id'].map(team_name)
    hz['own_gf6'] = hz['team'].map(tstate['gf6']); hz['own_ga6'] = hz['team'].map(tstate['ga6'])
    hz['opp_gf6'] = hz['opp'].map(tstate['gf6']);  hz['opp_ga6'] = hz['opp'].map(tstate['ga6'])
    hz['rest_days'] = 6
    hz['pos_code'] = hz['position'].map({'GKP':0,'DEF':1,'MID':2,'FWD':3})
    hz['value'] = hz['now_cost']

    tx_state = tx.sort_values('kickoff_time').groupby('team').tail(1).set_index('team')[TXCOLS]
    for c in TXCOLS:
        hz[f'own_{c}'] = hz['team'].map(tx_state[c])
        hz[f'opp_{c}'] = hz['opp'].map(tx_state[c])
    hz2 = add_derived(hz)
    hz2[FEATURES2] = hz2[FEATURES2].replace([np.inf, -np.inf], np.nan)
    return hz2, horizon, last_gw, calendar


# ---------------------------------------------------------------- Main
def run():
    t0 = time.time()
    D = load_data()
    players, next_gw, hist = D['players'], D['next_gw'], D['hist']

    df, tm = build_features(hist)
    tx = build_tx(hist)
    df2 = add_team_xg(df, tx).sort_values(['pkey','kickoff_time','fixture']).reset_index(drop=True)
    df2 = add_derived(df2)
    df2[FEATURES2] = df2[FEATURES2].replace([np.inf, -np.inf], np.nan)
    df2 = add_targets(df2)
    print('Features prêtes :', len(FEATURES2), '| lignes :', len(df2), f'| {time.time()-t0:.0f}s')

    train_all = df2[df2['total_points'].notna() & (df2['games_seen'] >= 3)]
    ref = train_all['kickoff_time'].max()
    comp_final = {c: fit_pair(train_all.dropna(subset=[c]), c, ref, n=250) for c in COMPS}
    tot_final = fit_pair(train_all, 'total_points', ref, n=400)
    HAS_Q = True
    try:    # modèle quantile : "plafond" probable (85e centile) pour le capitaine
        q_final = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.85, n_estimators=300, **P_XGB) \
                    .fit(train_all[FEATURES2], train_all['total_points'], sample_weight=recency(train_all, ref))
    except Exception as e:
        HAS_Q = False
        print('Quantile indisponible :', e)
    print(f'Modèles entraînés | {time.time()-t0:.0f}s')

    hz2, horizon, last_gw, calendar = build_horizon_rows(D, hist, tm, tx)
    comp_pred = sum(comp_final[c](hz2) for c in COMPS)
    hz2['pred2'] = (W_COMP * comp_pred + (1 - W_COMP) * tot_final(hz2)).clip(min=-1)
    if HAS_Q:
        hz2['pred_up'] = np.maximum(q_final.predict(hz2[FEATURES2]), hz2['pred2'])

    # Disponibilité : vrai % pour la prochaine GW, approximation ensuite
    fill = pd.Series(np.where(hz2['status'] == 'a', 100, 0), index=hz2.index)
    c_next = hz2['chance_of_playing_next_round'].fillna(fill) / 100
    later = pd.Series(np.where(hz2['status'].isin(['a','d']), 1.0,
                      np.where(hz2['status'].isin(['i','s']), 0.5, 0.0)), index=hz2.index)
    hz2['avail'] = np.where(hz2['event'] == next_gw, c_next, later)
    hz2['xp'] = hz2['pred2'] * hz2['avail']
    if HAS_Q:
        hz2['xp_up'] = hz2['pred_up'] * hz2['avail']

    def build_ev(col):
        m = (hz2.groupby(['element','event'])[col].sum().unstack(fill_value=0)
                .reindex(index=players['id'], columns=horizon, fill_value=0))
        return players.set_index('id')[['web_name','position','team','team_name','now_cost']].join(m)

    ev = build_ev('xp')
    ev_up = build_ev('xp_up') if HAS_Q else None
    tips, K = compute_tips_df(ev, ev_up, hz2, players, horizon, next_gw)

    art = dict(
        next_gw=next_gw, horizon=horizon, LAST_GW=last_gw, ev=ev, ev_up=ev_up, calendar=calendar,
        tips=tips, K=K,
        players=players[['id','web_name','position','team_name','now_cost','status',
                         'chance_of_playing_next_round','news','selected_by_percent']],
        events=D['events'][['id','deadline_time','is_current','is_next','average_entry_score']],
        team_name=D['team_name'], pts_gw=D['pts_gw'], first_value=D['first_value'],
        trained_at=pd.Timestamp.now(tz='UTC').isoformat(),
    )
    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(art, OUT_DIR / f'gw{next_gw}.pkl')
    joblib.dump(art, OUT_DIR / 'latest.pkl')
    print(f'✅ Sauvegardé data/latest.pkl (GW{next_gw}) en {time.time()-t0:.0f}s')


if __name__ == '__main__':
    run()