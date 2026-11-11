"""Headless smoke tests for the pygame front end."""
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import pygame as pg
import pytest

from escape import ALIEN, ROVER, SOFT
from escape.graph import manhattan
from ui.app import App


def key(k):
    return pg.event.Event(pg.KEYDOWN, key=k, mod=0, unicode='')


def click(pos, button=1):
    return pg.event.Event(pg.MOUSEBUTTONDOWN, pos=pos, button=button)


def test_menu_cycles_and_starts():
    app = App(seed=5)
    app.handle(key(pg.K_RIGHT))                 # mode -> You are the Alien
    assert app.mode == 'alien'
    app.handle(key(pg.K_DOWN))
    app.handle(key(pg.K_RIGHT))                 # rover AI changes
    app.handle(key(pg.K_RETURN))
    assert app.screen_name == 'game' and app.alien_ai is None
    app.draw()


@pytest.mark.parametrize('rover,alien', [('Minimax', 'MCTS'), ('Fuzzy', 'Core'), ('D* Lite', 'Greedy')])
def test_ai_vs_ai_plays_to_the_end(rover, alien):
    app = App('ai', rover, alien, seed=21)
    app.start_game()
    for _ in range(500):
        if app.state.result:
            break
        app.next_time = 0
        app.update()
        app.draw()
    assert app.state.result in (ROVER, ALIEN)
    app.draw()


def test_human_rover_moves_and_shoots():
    app = App('rover', alien='Greedy', seed=8)
    app.start_game()
    s = app.state
    tgt = next(v for v in s.map.nbrs[s.rover])
    d = (tgt[0] - s.rover[0], tgt[1] - s.rover[1])
    keys = {(-1, 0): pg.K_UP, (1, 0): pg.K_DOWN, (0, -1): pg.K_LEFT, (0, 1): pg.K_RIGHT}
    s.blk[tgt] = SOFT
    app.handle(key(keys[d]))                    # bump into soft boulder -> shoot
    assert s.shots == 1 and tgt not in s.blk
    app.handle(key(keys[d]))                    # now drive onto it
    assert s.rover == tgt
    app.handle(key(pg.K_SPACE))                 # wait
    assert s.tick == 3 and s.turn == ALIEN
    app.next_time = 0
    app.update()                                # AI alien strikes
    assert s.turn == ROVER
    app.draw()


def test_human_alien_clicks():
    app = App('alien', rover='A*', seed=8)
    app.start_game()
    s = app.state
    while s.turn != ALIEN:
        app.next_time = 0
        app.update()
    p = next(q for q in s.map.open_tiles if s.alien_can_target(q) and manhattan(q, s.rover) > 4)
    x, y = app.tile_center(p)
    app.handle(click((x, y), 3))                # right-click: debris
    assert p in s.pending and s.turn == ROVER
    while s.turn != ALIEN:
        app.next_time = 0
        app.update()
    budget = s.budget
    app.handle(click(app.tile_center(s.rover)))  # illegal: on the rover
    assert s.turn == ALIEN and s.budget == budget and app.flash
    app.handle(key(pg.K_SPACE))                 # hold
    assert s.turn == ROVER and s.budget == budget
    app.draw()
