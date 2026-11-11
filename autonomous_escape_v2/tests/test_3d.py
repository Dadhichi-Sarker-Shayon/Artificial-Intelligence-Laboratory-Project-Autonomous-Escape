"""Smoke tests for the 3D front end, rendered offscreen (needs a GPU)."""
import pytest

panda3d = pytest.importorskip('panda3d')

from escape import ALIEN, ROVER, SOFT


@pytest.fixture(scope='module')
def game():
    from ui3d.app3d import Game, configure
    configure(offscreen=True)
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
    g = Game(base, 'rover', alien='Greedy', seed=8)
    yield g
    base.destroy()


def frame(g, n=1):
    for _ in range(n):
        g.base.taskMgr.step()


def test_menu_then_ai_game_plays_to_the_end(game):
    g = game
    frame(g, 2)
    g.mode, g.rover_name, g.alien_name, g.speed = 'ai', 'A*', 'Core', 8
    for cam in ('Chase', 'Pilot', 'Overhead'):
        g.camera_mode = cam
        g.start_game()
        frame(g, 3)
    for _ in range(3000):
        if g.state.result:
            break
        g.next_time = 0
        g.anim = g.anim[:2] + (0.0, 1e-3) + g.anim[4:]
        frame(g)
    assert g.state.result in (ROVER, ALIEN)
    frame(g, 2)


def test_pilot_controls_turn_drive_and_shoot(game):
    g = game
    g.mode, g.alien_name, g.camera_mode = 'rover', 'Greedy', 'Pilot'
    g.start_game()
    frame(g, 2)
    s = g.state
    g.heading = next((v[0] - s.rover[0], v[1] - s.rover[1]) for v in s.map.nbrs[s.rover])
    ahead = (s.rover[0] + g.heading[0], s.rover[1] + g.heading[1])
    tick = s.tick
    g.anim = g.anim[:2] + (0.0, 1e-3) + g.anim[4:]
    g.on_key('arrow_left')                      # turning is free
    assert s.tick == tick
    g.anim = g.anim[:2] + (0.0, 1e-3) + g.anim[4:]
    g.on_key('arrow_right')
    s.blk[ahead] = SOFT
    g.anim = g.anim[:2] + (0.0, 1e-3) + g.anim[4:]
    g.on_key('arrow_up')                        # bump: laser
    assert s.shots == 1 and ahead not in s.blk
    g.anim = g.anim[:2] + (0.0, 1e-3) + g.anim[4:]
    g.on_key('arrow_up')                        # drive
    assert s.rover == ahead
    frame(g, 3)
