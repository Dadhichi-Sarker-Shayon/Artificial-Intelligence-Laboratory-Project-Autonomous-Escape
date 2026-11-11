"""Smoke tests for the v3 3D front end, rendered offscreen (needs a GPU)."""
import pytest

panda3d = pytest.importorskip('panda3d')

from escape import ALIEN, ROVER, SOFT


@pytest.fixture(scope='module')
def game(tmp_path_factory):
    from ui3d import story
    story.SAVE = str(tmp_path_factory.mktemp('save') / 'save.json')     # never touch the real save
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


def unbusy(g):
    g.anim = g.anim[:2] + (0.0, 1e-3) + g.anim[4:]


def test_menus_navigate(game):
    g = game
    frame(g, 2)
    assert g.screen == 'title'
    g.on_key('enter')                           # campaign
    assert g.screen == 'missions'
    g.on_key('enter')                           # first mission briefing
    assert g.screen == 'briefing'
    frame(g, 3)
    g.on_key('escape')
    g.on_key('escape')
    assert g.screen == 'title'
    g.show_setup()
    frame(g, 2)
    g.show_help()
    frame(g, 2)


def test_ai_game_plays_to_debrief(game):
    g = game
    g.mode, g.rover_name, g.alien_name, g.speed = 'ai', 'A*', 'Core', 8
    for cam in ('Chase', 'Pilot', 'Overhead'):
        g.camera_mode = cam
        g.start_free()
        frame(g, 3)
    g.on_key('space')                           # skip the intro
    assert g.phase == 'play'
    for _ in range(3000):
        if g.state.result:
            break
        g.next_time = 0
        unbusy(g)
        frame(g)
    assert g.state.result in (ROVER, ALIEN)
    frame(g, 3)
    assert g.phase == 'outro'
    g.on_key('space')
    assert g.phase == 'debrief'
    frame(g, 2)


def test_mission_controls_pulse_drive_shoot_autopilot(game):
    g = game
    g.camera_mode = None                        # missions default to the pilot camera
    g.start_mission(1)                          # mission 2: radar pulse online
    frame(g, 2)
    g.on_key('space')                           # skip intro
    s = g.state
    unbusy(g)
    f = s.fuel
    g.on_key('q')
    assert s.scans == 1 and s.fuel == f - s.rules.scan_cost
    while s.turn == ALIEN or s.result:
        g.next_time = 0
        frame(g)
    g.heading = next((v[0] - s.rover[0], v[1] - s.rover[1]) for v in s.map.nbrs[s.rover])
    ahead = (s.rover[0] + g.heading[0], s.rover[1] + g.heading[1])
    unbusy(g)
    tick = s.tick
    g.on_key('arrow_left')                      # turning is free
    assert s.tick == tick
    unbusy(g)
    g.on_key('arrow_right')
    s.blk[ahead] = SOFT
    unbusy(g)
    g.on_key('arrow_up')                        # bump: laser
    assert s.shots == 1 and ahead not in s.blk
    while s.turn == ALIEN:
        g.next_time = 0
        frame(g)
    unbusy(g)
    g.on_key('arrow_up')                        # drive
    assert s.rover == ahead
    g.waypoint = s.map.goal                     # autopilot takes over
    for _ in range(40):
        g.next_time = 0
        unbusy(g)
        frame(g)
    assert s.tick > tick + 3
    g.restart()
    g.on_key('space')                           # skip intro
    g.on_key('escape')                          # pause menu
    assert g.paused and g.overlay is not None
    g.on_key('escape')
    assert not g.paused
    frame(g, 3)
