"""Run the AI-vs-AI benchmark.

Examples:
  python bench.py                                  # all rovers x all aliens, seeds 100-299
  python bench.py --seeds 100-139 --aliens Core,Greedy
  python bench.py --mcts-iters 80                  # cheaper MCTS alien
"""
import argparse
import os
import time

from escape.agents import ROVER_BY_NAME, ALIEN_BY_NAME
from escape.bench import run, summarize, heatmap


def parse_seeds(text):
    a, _, b = text.partition('-')
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', default='100-299', help='held-out maps, e.g. 100-299')
    ap.add_argument('--rovers', default='all', help='comma list: ' + ', '.join(ROVER_BY_NAME))
    ap.add_argument('--aliens', default='all', help='comma list: ' + ', '.join(ALIEN_BY_NAME))
    ap.add_argument('--jobs', type=int, default=0, help='worker processes (default: CPUs - 1)')
    ap.add_argument('--mcts-iters', type=int, default=150)
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    rovers = list(ROVER_BY_NAME) if args.rovers == 'all' else args.rovers.split(',')
    aliens = list(ALIEN_BY_NAME) if args.aliens == 'all' else args.aliens.split(',')
    seeds = parse_seeds(args.seeds)
    alien_kw = {'MCTS': {'iterations': args.mcts_iters, 'time_limit': 10.0}}
    print(f'{len(rovers)} rovers x {len(aliens)} aliens x {len(seeds)} maps '
          f'= {len(rovers) * len(aliens) * len(seeds)} games')
    t0 = time.time()
    rows = run(seeds, rovers, aliens, jobs=args.jobs or None, alien_kw=alien_kw)
    os.makedirs(args.out, exist_ok=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'games.csv'), index=False)
    summary = summarize(rows)
    summary.round(2).to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    heatmap(summary, os.path.join(args.out, 'alien_win_heatmap.png'))

    cols = ['rover', 'alien', 'alien_win', 'ci_low', 'ci_high', 'sealed', 'escape_score',
            'shots', 'rover_ms', 'alien_ms', 'rover_nodes']
    print(summary[cols].round(1).to_string(index=False))
    print(f'\nDone in {time.time() - t0:.0f}s. Results in {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
