"""Section 15 du notebook : tips globaux (identiques pour tous les utilisateurs d'une même GW)."""
import numpy as np
import pandas as pd


def compute_tips_df(ev, ev_up, hz2, players, horizon, next_gw):
    """Appelé par train_job.py. Retourne (tableau des tips indexé par id joueur, nom de la colonne xPts sur N GW)."""
    INFO = players.set_index('id')
    gws4 = horizon[:4]
    K = f'xPts_{len(gws4)}GW'
    adv = (hz2[hz2['event'] == next_gw]
           .assign(a=lambda d: d['opp'] + np.where(d['home'] == 1, ' (D)', ' (E)'))
           .groupby('element')['a'].apply(', '.join))

    tips = ev[['web_name', 'position', 'team_name', 'now_cost']].copy()
    tips['prix'] = tips['now_cost'] / 10
    tips['xPts_GW'] = ev[next_gw].round(2)
    tips[K] = ev[gws4].sum(axis=1).round(1)
    if ev_up is not None:
        tips['plafond'] = ev_up[next_gw].round(1)
    tips['adversaires'] = adv.reindex(tips.index)
    tips['selection_%'] = INFO['selected_by_percent'].astype(float)
    tips['penalty'] = INFO['penalties_order'].eq(1).map({True: 'P', False: ''})
    tips['ep_FPL'] = pd.to_numeric(INFO['ep_next'], errors='coerce')
    tips['net_transferts'] = INFO['transfers_in_event'] - INFO['transfers_out_event']
    tips['news'] = INFO['news']
    tips['status'] = INFO['status']
    tips['valeur'] = (tips[K] / tips['prix']).round(2)
    return tips, K


def _rows(df, cols):
    return [{k: row[v] for k, v in cols.items()} for _, row in df.iterrows()]


def build_tips(art):
    """Appelé par l'API : met les tips en forme pour le front (même structure que demo() du front)."""
    t, K = art['tips'], art['K']
    base = {'name': 'web_name', 'team': 'team_name', 'price': 'prix'}

    cap_cols = {**base, 'xpts': 'xPts_GW', 'vs': 'adversaires', 'sel': 'selection_%'}
    if 'plafond' in t.columns:
        cap_cols['ceiling'] = 'plafond'
    captain = _rows(t.nlargest(5, 'xPts_GW'), cap_cols)

    differentials = _rows(t[t['selection_%'] < 8].nlargest(6, K), {**base, 'pts4': K, 'sel': 'selection_%'})

    value = {pos: _rows(t[(t['position'] == pos) & (t[K] > 0)].nlargest(5, 'valeur'),
                        {**base, 'xpts': K, 'valeur': 'valeur'})
             for pos in ('GKP', 'DEF', 'MID', 'FWD')}

    hot = t[t[K] > t[K].quantile(0.9)]
    rises = _rows(hot.nlargest(6, 'net_transferts'), {'name': 'web_name', 'price': 'prix', 'pts4': K, 'net': 'net_transferts'})
    falls = _rows(t[t['selection_%'] >= 5].nsmallest(6, 'net_transferts'),
                  {'name': 'web_name', 'price': 'prix', 'net': 'net_transferts'})

    teams = {}
    for grp, poss in (('attack', ['MID', 'FWD']), ('defense', ['GKP', 'DEF'])):
        s = (t[t['position'].isin(poss)].sort_values(K, ascending=False)
              .groupby('team_name').head(3).groupby('team_name')[K].sum()
              .sort_values(ascending=False).head(8))
        teams[grp] = [{'team': k, 'score': round(float(v), 1)} for k, v in s.items()]

    return {'captain': captain, 'differentials': differentials, 'value': value,
            'rises': rises, 'falls': falls, 'teams': teams}