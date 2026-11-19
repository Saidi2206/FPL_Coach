"""Optimiseur de squad (section 7 du notebook). Ne dépend que de la matrice `ev` : aucune variable globale."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

POS_NEED = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
FORMS = [(3,4,3),(3,5,2),(4,3,3),(4,4,2),(4,5,1),(5,3,2),(5,4,1),(5,2,3)]


def best_xi(sub, g):
    """Meilleur XI valide pour la GW g -> (points sans capitaine, ids du XI, id du capitaine)."""
    by = {p: sub[sub['position'] == p].sort_values(g, ascending=False) for p in POS_NEED}
    best_ = (-1, None)
    for d, m, f in FORMS:
        if len(by['GKP']) < 1 or len(by['DEF']) < d or len(by['MID']) < m or len(by['FWD']) < f:
            continue
        ids = list(by['GKP'].index[:1]) + list(by['DEF'].index[:d]) + \
              list(by['MID'].index[:m]) + list(by['FWD'].index[:f])
        pts = sub.loc[ids, g].sum()
        if pts > best_[0]:
            best_ = (pts, ids)
    if best_[1] is None:
        return 0.0, [], None
    return best_[0], best_[1], sub.loc[best_[1], g].idxmax()


def best_xi_pts(sub, g):
    pts, ids, cap = best_xi(sub, g)
    return pts + (sub.loc[cap, g] if cap is not None else 0)


def make_pool(ev, gws):
    d = ev.copy()
    d['tot'] = d[gws].sum(axis=1)
    d['val'] = d['tot'] / (d['now_cost'] / 10)
    keep = set()
    for pos, k in (('GKP', 12), ('DEF', 40), ('MID', 40), ('FWD', 25)):
        s = d[d['position'] == pos]
        keep |= set(s.nlargest(k, 'tot').index) | set(s.nlargest(12, 'val').index) \
              | set(s[s['tot'] > 0].nsmallest(8, 'now_cost').index)
    return d.loc[sorted(keep)]


def solve_plan(ev, gws, budget, bb_gw=None, bench_w=0.1, decay=0.92, time_limit=40,
               owned=None, max_tr=None, cost=None):
    """Squad de 15 optimal sur les GW `gws` (XI + capitaine par GW ; Bench Boost si bb_gw).
       owned / max_tr : on part de ton équipe et on limite les transferts ; cost : prix de vente."""
    pool = make_pool(ev, gws)
    owned_set = set(owned or [])
    missing = [i for i in owned_set if i not in pool.index]
    if missing:
        import pandas as pd
        pool = pd.concat([pool, ev.loc[missing]])
    ids = list(pool.index)
    n, G = len(ids), len(gws)
    pos_of, team_of = pool['position'].to_dict(), pool['team'].to_dict()
    cost_of = pool['now_cost'].to_dict()
    if cost:
        for i, v in cost.items():
            if i in cost_of:
                cost_of[i] = v
    w = {g: decay ** k for k, g in enumerate(gws)}

    NV = n + 2 * G * n
    S = lambda gi, k: n + gi * n + k
    C = lambda gi, k: n + G * n + gi * n + k

    cvec = np.zeros(NV); ub = np.ones(NV)
    for gi, g in enumerate(gws):
        e = pool[g].to_numpy(dtype=float) * w[g]
        for k in range(n):
            if g == bb_gw:
                cvec[k] -= e[k]; cvec[C(gi, k)] -= e[k]
            else:
                cvec[k] -= e[k] * bench_w
                cvec[S(gi, k)] -= e[k] * (1 - bench_w)
                cvec[C(gi, k)] -= e[k]
        if g == bb_gw:
            for k in range(n):
                ub[S(gi, k)] = 0

    rows, lbs, ubs = [], [], []
    def add(coefs, lo, hi):
        rows.append(coefs); lbs.append(lo); ubs.append(hi)

    add({k: 1 for k in range(n)}, 15, 15)
    for pos, cnt_ in POS_NEED.items():
        add({k: 1 for k, i in enumerate(ids) if pos_of[i] == pos}, cnt_, cnt_)
    add({k: float(cost_of[i]) for k, i in enumerate(ids)}, -np.inf, budget)
    for t in set(team_of.values()):
        add({k: 1 for k, i in enumerate(ids) if team_of[i] == t}, -np.inf, 3)
    if max_tr is not None and owned_set:
        add({k: 1 for k, i in enumerate(ids) if i in owned_set}, 15 - max_tr, 15)

    for gi, g in enumerate(gws):
        add({C(gi, k): 1 for k in range(n)}, 1, 1)
        if g == bb_gw:
            for k in range(n):
                add({C(gi, k): 1, k: -1}, -np.inf, 0)
        else:
            add({S(gi, k): 1 for k in range(n)}, 11, 11)
            for k in range(n):
                add({S(gi, k): 1, k: -1}, -np.inf, 0)
                add({C(gi, k): 1, S(gi, k): -1}, -np.inf, 0)
            for pos, (lo, hi) in {'GKP': (1, 1), 'DEF': (3, 5), 'MID': (2, 5), 'FWD': (1, 3)}.items():
                add({S(gi, k): 1 for k, i in enumerate(ids) if pos_of[i] == pos}, lo, hi)

    A = lil_matrix((len(rows), NV))
    for r_, coefs in enumerate(rows):
        for col, v in coefs.items():
            A[r_, col] = v

    res_ = milp(c=cvec, constraints=LinearConstraint(A.tocsr(), np.array(lbs), np.array(ubs)),
                integrality=np.ones(NV), bounds=Bounds(np.zeros(NV), ub),
                options={'time_limit': time_limit, 'disp': False})
    if res_.x is None:
        raise RuntimeError(f'Pas de solution : {res_.message} (vérifie le budget)')
    chosen = [ids[k] for k in range(n) if res_.x[k] > 0.5]
    if len(chosen) != 15:
        raise RuntimeError(f'Squad de {len(chosen)} joueurs au lieu de 15 : {res_.message}')
    return ev.loc[chosen]


def plan_score(sq, gws, bench_w=0.1):
    tot = 0
    for g in gws:
        pts, _, cap = best_xi(sq, g)
        tot += pts + sq.loc[cap, g] + bench_w * (sq[g].sum() - pts)
    return tot