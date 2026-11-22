"""Benchmark: every rover against every alien on the same seeded maps."""
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

from .config import DEFAULT
from .state import GameState, ALIEN, ROVER
from .agents import make_rover, make_alien


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def play(seed, rover_name, alien_name, rules=DEFAULT, alien_kw=None):
    """Play one full game; returns a result row."""
    s = GameState.new(seed, rules)
    rover = make_rover(rover_name, seed)
    alien = make_alien(alien_name, seed, **(alien_kw or {}))
    rover.reset()
    alien.reset()
    t_r = t_a = 0.0
    n_r = n_a = 0
    while s.result is None:
        if s.turn == ALIEN:
            t0 = time.perf_counter()
            a = alien.act(s)
            t_a += time.perf_counter() - t0
            n_a += 1
            s.apply_alien(a)
        else:
            v = s.rover_view()
            t0 = time.perf_counter()
            a = rover.act(v)
            t_r += time.perf_counter() - t0
            n_r += 1
            s.apply_rover(a)
    return dict(seed=seed, rover=rover_name, alien=alien_name, winner=s.result,
                reason=s.reason, score=round(s.score, 2), turns=s.tick, shots=s.shots,
                debris_hits=s.hits, alien_actions=s.alien_moves,
                rover_ms=1000 * t_r / max(n_r, 1), alien_ms=1000 * t_a / max(n_a, 1),
                rover_nodes=rover.nodes / max(n_r, 1), alien_nodes=alien.nodes / max(n_a, 1))


def _play_args(args):
    return play(*args)


def run(seeds, rovers, aliens, jobs=None, rules=DEFAULT, alien_kw=None, progress=print):
    tasks = [(s, r, a, rules, (alien_kw or {}).get(a)) for r in rovers for a in aliens for s in seeds]
    rows = []
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    if jobs == 1:
        for i, t in enumerate(tasks, 1):
            rows.append(play(*t))
            if progress and i % 50 == 0:
                progress(f'  {i}/{len(tasks)} games, {time.time() - t0:.0f}s')
    else:
        with ProcessPoolExecutor(jobs) as ex:
            for i, row in enumerate(ex.map(_play_args, tasks, chunksize=4), 1):
                rows.append(row)
                if progress and i % 100 == 0:
                    progress(f'  {i}/{len(tasks)} games, {time.time() - t0:.0f}s')
    return rows


def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    out = []
    for (r, a), g in df.groupby(['rover', 'alien'], sort=False):
        n = len(g)
        k = int((g.winner == ALIEN).sum())
        lo, hi = wilson(k, n)
        esc = g[g.winner == ROVER]
        out.append(dict(rover=r, alien=a, games=n, alien_win=100 * k / n,
                        ci_low=100 * lo, ci_high=100 * hi,
                        sealed=100 * (g.reason == 'Every route sealed').mean(),
                        escape_score=esc.score.mean() if len(esc) else 0.0,
                        shots=g.shots.mean(), debris_hits=g.debris_hits.mean(),
                        rover_ms=g.rover_ms.mean(), alien_ms=g.alien_ms.mean(),
                        rover_nodes=g.rover_nodes.mean(), alien_nodes=g.alien_nodes.mean()))
    return pd.DataFrame(out)


def heatmap(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pv = summary.pivot(index='rover', columns='alien', values='alien_win')
    pv = pv.loc[list(dict.fromkeys(summary.rover)), list(dict.fromkeys(summary.alien))]
    fig, ax = plt.subplots(figsize=(2.4 + 1.3 * pv.shape[1], 1.4 + 0.6 * pv.shape[0]))
    im = ax.imshow(pv.values, cmap='magma_r', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    ax.set_xlabel('Alien')
    ax.set_ylabel('Rover')
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                    color='white' if v > 55 else 'black', fontsize=10)
    ax.set_title('Alien win rate')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
