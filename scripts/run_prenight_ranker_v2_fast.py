from __future__ import annotations

import run_prenight_ranker_v2 as ranker

_original = ranker._greedy_diverse


def _bounded_greedy(x, n):
    # Ranking target is at most Top500. Searching the first 4000 scored slots gives
    # an 8x buffer while avoiding quadratic scans through the full road-time grid.
    return _original(x.head(4000).copy(), n)


ranker._greedy_diverse = _bounded_greedy

if __name__ == '__main__':
    ranker.main()
