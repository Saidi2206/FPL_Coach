"""Partie « par ID » (sections 11 à 15 du notebook). Utilise les prédictions déjà calculées (art['ev'])."""
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import requests

from .optimizer import POS_NEED, best_xi, best_xi_pts, plan_score, solve_plan
from .tips import build_tips

API = 'https://fantasy.premierleague.com/api'
HEADERS = {'User-Agent': 'Mozilla/5.0'}
N_GW_TRANSFERS = 4      # horizon (en GW) pour juger un transfert
TL_TRANSFERS = 15       # secondes max par optimisation de transferts
TL_CHIPS = 10           # secondes max par optimisation de chips


class TeamNotFound(Exception):
    pass


class UpstreamError(Exception):
    pass


def to_py(o):
    """numpy / pandas -> types JSON (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (o != o or o in (float('inf'), float('-inf'))):
        return None
    return o


def fpl_get(path):
    last = None
    for _ in range(3):
        try:
            r = requests.get(f'{API}/{path}', timeout=20, headers=HEADERS)
        except requests.RequestException as e:
            last = e; time.sleep(1); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            raise TeamNotFound(path)
        last = r.status_code; time.sleep(1)
    raise UpstreamError(f'API FPL indisponible ({last}). Réessaie dans une minute.')


@lru_cache(maxsize=4096)
def get_picks(team_id, gw):          # les picks d'une GW passée ne changent plus : on les garde en mémoire
    return fpl_get(f'entry/{team_id}/event/{gw}/picks/')['picks']


def team_context(team_id, art, ft_override=None):
    """Équipe actuelle, banque, transferts gratuits, prix de vente (sections 6 et 11)."""
    events, players, next_gw = art['events'], art['players'], art['next_gw']
    cur = int(events.loc[events['is_current'], 'id'].iloc[0]) if events['is_current'].any() else next_gw - 1
    if cur < 1:
        raise UpstreamError('La saison n’a pas commencé : aucune équipe à analyser.')

    entry = fpl_get(f'entry/{team_id}/')
    hist_me = fpl_get(f'entry/{team_id}/history/')
    transfers = fpl_get(f'entry/{team_id}/transfers/')
    pk = fpl_get(f'entry/{team_id}/event/{cur}/picks/')

    gw_info = {r['event']: r for r in hist_me['current']}
    chip_gw = {c['event']: c['name'] for c in hist_me['chips']}
    my_ids = [p['element'] for p in pk['picks']]
    bank = pk['entry_history']['bank']
    pending = [t for t in transfers if t['event'] == next_gw]       # transferts déjà faits pour la prochaine GW
    for t in pending:
        if t['element_out'] in my_ids:
            my_ids[my_ids.index(t['element_out'])] = t['element_in']
            bank += t['element_out_cost'] - t['element_in_cost']

    ft = 1      # estimation : 1 en GW2, +1 par GW non utilisé, max 5 ; WC/FH n'en consomment pas
    for g in range(2, cur + 1):
        r = gw_info.get(g)
        if r is None:
            continue
        used = 0 if chip_gw.get(g) in ('wildcard', 'freehit') else r['event_transfers']
        ft = min(max(ft - used, 0) + 1, 5)
    FT = ft_override if ft_override is not None else ft
    FT_left = max(FT - len(pending), 0)

    buy = dict(art['first_value'])           # prix d'achat ; sinon dernier transfert entrant
    for t in sorted(transfers, key=lambda t: t['event']):
        buy[t['element_in']] = t['element_in_cost']
    now = players.set_index('id')['now_cost'].to_dict()

    def sell(i):                              # prix de vente = achat + moitié du profit (arrondi vers le bas)
        b, n_ = buy.get(i, now[i]), now[i]
        return n_ if n_ <= b else b + (n_ - b) // 2
    sell_price = {i: sell(i) for i in my_ids}

    return dict(cur=cur, entry=entry, hist_me=hist_me, gw_info=gw_info, my_ids=my_ids, bank=bank,
                pending=len(pending), FT_left=FT_left, sell_price=sell_price,
                total_budget=bank + sum(sell_price.values()), chips_used=hist_me['chips'])


def _pl(ev, i, gws, price=None):
    r = ev.loc[i]
    return {'name': r['web_name'], 'pos': r['position'], 'team': r['team_name'],
            'xpts': round(float(r[gws].sum()), 1),
            'price': (price if price is not None else r['now_cost']) / 10}


def analyze(team_id, art):
    """Réponse principale (rapide, quelques secondes) : équipe, XI, transferts, tips, historique."""
    ctx = team_context(team_id, art)
    ev, players, events = art['ev'], art['players'], art['events']
    horizon, next_gw = art['horizon'], art['next_gw']
    my_ids, sell_price, FT_left = ctx['my_ids'], ctx['sell_price'], ctx['FT_left']
    gws_t = horizon[:N_GW_TRANSFERS]
    mine = ev.loc[my_ids]

    # ---- Transferts : gain net pour 0 à 4 transferts
    base = plan_score(mine, gws_t)
    rows, sols = [], {}
    for k in range(5):
        sq = solve_plan(ev, gws_t, ctx['total_budget'], decay=1.0, owned=my_ids, max_tr=k,
                        cost=sell_price, time_limit=TL_TRANSFERS)
        n_tr = len(set(sq.index) - set(my_ids))
        hit = 4 * max(0, n_tr - FT_left)
        rows.append({'max': k, 'made': n_tr, 'hit': hit, 'net': round(float(plan_score(sq, gws_t) - base - hit), 1)})
        sols[k] = sq
    best = max(rows, key=lambda r: r['net'])
    if best['net'] <= 0:
        best = rows[0]                    # aucun transfert ne vaut le coup
    sq = sols[best['max']]

    outs = ev.loc[sorted(set(my_ids) - set(sq.index))]
    ins = ev.loc[sorted(set(sq.index) - set(my_ids))]
    advice = []
    for pos in POS_NEED:
        o, i_ = outs[outs['position'] == pos], ins[ins['position'] == pos]
        for (oi, _), (ii, _) in zip(o.iterrows(), i_.iterrows()):
            advice.append({'pos': pos, 'out': _pl(ev, oi, gws_t, sell_price[oi]), 'in': _pl(ev, ii, gws_t)})

    # ---- XI, capitaine, banc (équipe après les transferts conseillés)
    _, xi, cap = best_xi(sq, next_gw)
    xi_sorted = sq.loc[xi].sort_values(next_gw, ascending=False)
    vice = xi_sorted.index[1] if xi_sorted.index[0] == cap else xi_sorted.index[0]
    bench = sq.drop(index=xi).copy()
    bench['_gk'] = (bench['position'] == 'GKP').astype(int)
    bench = bench.sort_values(['_gk', next_gw], ascending=[False, False])     # GK remplaçant en 1er
    order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    xi_list = [_pl(ev, i, [next_gw]) for i in sorted(xi, key=lambda i: order[ev.loc[i, 'position']])]
    lineup = {'captain': ev.loc[cap, 'web_name'], 'vice': ev.loc[vice, 'web_name'], 'xi': xi_list,
              'bench': [_pl(ev, i, [next_gw]) for i in bench.index]}

    # ---- Historique + regret capitaine (picks des GW passées en parallèle)
    gw_info, name_of, pts_gw = ctx['gw_info'], players.set_index('id')['web_name'].to_dict(), art['pts_gw']

    def regret(g):
        try:
            pg = get_picks(team_id, g)
            cp = next(p for p in pg if p['is_captain'])
            bp = max((p for p in pg if p['multiplier'] > 0), key=lambda p: pts_gw.get((p['element'], g), 0))
            return (pts_gw.get((bp['element'], g), 0) - pts_gw.get((cp['element'], g), 0)) * (cp['multiplier'] - 1)
        except Exception:
            return 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        captain_regret = sum(ex.map(regret, sorted(gw_info)))

    avg = events.set_index('id')['average_entry_score']
    gws_hist = [{'gw': g, 'points': r['points'], 'average': avg.get(g), 'rank': r['overall_rank']}
                for g, r in sorted(gw_info.items())]
    history = {'gws': gws_hist,
               'hits_lost': sum(r['event_transfers_cost'] for r in gw_info.values()),
               'bench_points': sum(r['points_on_bench'] for r in gw_info.values()),
               'captain_regret': captain_regret}

    # ---- Tips globaux + tips personnalisés
    tips = build_tips(art)
    t, K = art['tips'], art['K']
    mt = t.loc[my_ids]
    tips['mine'] = {
        'weakest': [{'name': r['web_name'], 'pos': r['position'], 'price': r['prix'], 'pts4': r[K], 'news': r['news']}
                    for _, r in mt.nsmallest(5, K).iterrows()],
        'leaving': [{'name': r['web_name'], 'price': r['prix'], 'net': r['net_transferts']}
                    for _, r in mt.nsmallest(3, 'net_transferts').iterrows()],
    }
    pl_ = players.set_index('id').loc[my_ids]
    alerts = [{'name': r['web_name'], 'status': r['status'], 'chance': r['chance_of_playing_next_round'], 'news': r['news']}
              for _, r in pl_[pl_['status'] != 'a'].iterrows()]

    entry = ctx['entry']
    return to_py({
        'gw': next_gw,
        'deadline': events.set_index('id')['deadline_time'][next_gw],
        'team': {'name': entry['name'], 'rank': entry.get('summary_overall_rank'),
                 'points': entry.get('summary_overall_points'), 'bank': ctx['bank'] / 10,
                 'free_transfers': FT_left},
        'lineup': lineup,
        'transfers': {'horizon': len(gws_t), 'gain_net': best['net'], 'hit': best['hit'],
                      'advice': advice, 'options': rows},
        'tips': tips, 'alerts': alerts, 'history': history,
    })


def chips_plan(team_id, art):
    """Plan Wildcard / Bench Boost / Free Hit (sections 12 à 14). Plus lent : endpoint séparé."""
    ctx = team_context(team_id, art)
    ev, events, calendar = art['ev'], art['events'], art['calendar']
    horizon, LAST_GW = art['horizon'], art['LAST_GW']
    mine, budget = ev.loc[ctx['my_ids']], ctx['total_budget']
    dl = events.set_index('id')['deadline_time']
    L = min(4, len(horizon))

    wc = []
    for w0 in horizon[:8]:
        gws = [g for g in horizon if w0 <= g < w0 + L]
        if len(gws) < L:
            continue
        sq = solve_plan(ev, gws, budget, time_limit=TL_CHIPS)
        wc.append({'gw': w0, 'gain': round(float(sum(best_xi_pts(sq, g) for g in gws) -
                                                  sum(best_xi_pts(mine, g) for g in gws)), 1)})

    fh = []
    for g in horizon:
        sq = solve_plan(ev, [g], budget, bench_w=0.05, time_limit=TL_CHIPS)
        fh.append({'gw': g, 'matchs': int(calendar.loc[g, 'matchs']),
                   'gain': round(float(best_xi_pts(sq, g) - best_xi_pts(mine, g)), 1)})

    bb = []
    cand = horizon[1:]
    if cand:
        top = calendar.loc[cand, 'matchs'].nlargest(4).index.tolist()
        for gb in sorted(set(top + horizon[1:2])):
            gws = [g for g in horizon if gb - 1 <= g <= gb + 1]
            sq = solve_plan(ev, gws, budget, bb_gw=gb, time_limit=TL_CHIPS)
            pts_xi, _, _ = best_xi(sq, gb)
            p_mine, _, _ = best_xi(mine, gb)
            bb.append({'gw': gb, 'bench': round(float(sq[gb].sum() - pts_xi), 1),
                       'bench_mine': round(float(mine[gb].sum() - p_mine), 1)})

    # Chips déjà jouées dans cette moitié de saison -> exclues du plan
    first_half = LAST_GW == 19
    used_names = {c['name'] for c in ctx['chips_used'] if (c['event'] <= 19) == first_half}
    plan, taken = [], set()
    if bb:
        b = max(bb, key=lambda r: r['bench']); gb = b['gw']
        if 'wildcard' not in used_names:
            plan.append({'chip': 'Wildcard', 'gw': gb - 1, 'deadline': dl[gb - 1], 'gain': 'prépare le Bench Boost'})
        if 'bboost' not in used_names:
            plan.append({'chip': 'Bench Boost', 'gw': gb, 'deadline': dl[gb], 'gain': f"+{b['bench']} pts depuis le banc"})
        taken = {gb, gb - 1}
    if 'freehit' not in used_names:
        ok = [r for r in fh if r['gw'] not in taken]
        if ok:
            f = max(ok, key=lambda r: r['gain'])
            plan.append({'chip': 'Free Hit', 'gw': f['gw'], 'deadline': dl[f['gw']], 'gain': f"+{f['gain']} pts"})
    plan.sort(key=lambda r: r['gw'])

    return to_py({
        'plan': plan, 'used': [{'name': c['name'], 'event': c['event']} for c in ctx['chips_used']],
        'note': 'Règle pratique : sous ~1,5 pt/GW de gain, tes transferts gratuits suffisent.',
        'wildcard': wc, 'freehit': fh, 'benchboost': bb,
    })