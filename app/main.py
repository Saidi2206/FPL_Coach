"""API FPL. Lancer depuis le dossier backend/ :  uvicorn app.main:app --reload"""
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .team import TeamNotFound, UpstreamError, analyze, chips_plan, to_py
from .tips import build_tips

DATA = Path(os.getenv('FPL_DATA', Path(__file__).parent.parent / 'data' / 'latest.pkl'))
ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        'CORS_ORIGINS',
        'http://localhost:4200'
    ).split(',')
]

app = FastAPI(title='FPL Coach API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=['GET'],
    allow_headers=['*']
)

_state = {'mtime': 0.0, 'art': None}
_cache = {}
_lock = threading.Lock()


def get_art():
    """Recharge automatiquement data/latest.pkl dès que train_job.py l'a mis à jour (nouvelle GW)."""
    if not DATA.exists():
        raise HTTPException(503, 'Aucune prédiction disponible : lance train_job.py.')
    m = DATA.stat().st_mtime
    if m != _state['mtime']:
        with _lock:
            if m != _state['mtime']:
                _state['art'] = joblib.load(DATA)
                _state['mtime'] = m
                _cache.clear()
    return _state['art']


def cached(key, fn):
    if key not in _cache:
        if len(_cache) > 2000:
            _cache.clear()
        _cache[key] = fn()
    return _cache[key]


def guarded(fn):
    try:
        return fn()
    except TeamNotFound:
        raise HTTPException(404, 'Équipe introuvable')
    except UpstreamError as e:
        raise HTTPException(502, str(e))
    except RuntimeError as e:        # solveur sans solution, etc.
        raise HTTPException(500, str(e))


@app.get('/api/health')
def health():
    return {'ok': True}


@app.get('/api/meta')
def meta():
    art = get_art()
    gw = art['next_gw']
    return to_py({'gw': gw, 'deadline': art['events'].set_index('id')['deadline_time'][gw],
                  'updated_at': datetime.fromtimestamp(_state['mtime'], timezone.utc).isoformat(),
                  'calendar': art['calendar'].reset_index().rename(columns={'index': 'gw'}).to_dict('records')})


@app.get('/api/tips')
def tips():
    art = get_art()
    return to_py(cached(('tips', art['next_gw']), lambda: build_tips(art)))


@app.get('/api/team/{team_id}')
def team(team_id: int):
    art = get_art()
    return guarded(lambda: cached((team_id, art['next_gw']), lambda: analyze(team_id, art)))


@app.get('/api/team/{team_id}/chips')
def team_chips(team_id: int):
    art = get_art()
    return guarded(lambda: cached((team_id, art['next_gw'], 'chips'), lambda: chips_plan(team_id, art)))