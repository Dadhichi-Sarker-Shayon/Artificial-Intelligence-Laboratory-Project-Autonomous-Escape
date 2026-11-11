"""Autonomous Escape v3: campaign, cockpit HUD, cinematics and sound on top
of the v2 3D pilot view. Same rules engine and AIs as versions 1 and 2.

Screens   title -> campaign (missions) -> briefing -> game -> debrief
                 -> free play setup -> game
Game      intro fly-in -> play -> outro (shuttle launch / signal lost) -> debrief
Cameras   Pilot (mast), Chase, Overhead. V cycles; right-drag looks / orbits,
          mouse wheel zooms.
Rover     W/S drive, A/D turn, Space wait, Q radar pulse, click the ground
          to send the autopilot there.
"""
import math
import os
import random
import sys

import numpy as np
from panda3d.core import (CardMaker, LineSegs, NodePath, Point3, TextNode, TransparencyAttrib, Vec3,
                          loadPrcFileData)

from escape import GameState, OPEN, SOFT, HARD, ROVER, ALIEN, INF, DEFAULT
from escape.agents import ROVER_BY_NAME, ALIEN_BY_NAME, make_rover, make_alien, CoreAlien
from escape.graph import astar, manhattan
from . import models, story
from .fx import Effects
from .motion import Motion, angle_diff
from .widgets import CYAN, DIM, GREEN, MUTED, ORANGE, PURPLE, RED, TEXT, YELLOW, UI, star

W, H = 1280, 780
MODES = [('ai', 'AI vs AI'), ('alien', 'You are the Alien'), ('rover', 'You are the Rover')]
CAMERAS = ['Pilot', 'Chase', 'Overhead']
SPEEDS = [1, 2, 4, 8]
METEOR_DIR = Vec3(0.45, 0.55, 0.72).normalized()     # debris falls in from the north-east sky
DRIVE_KEYS = ('arrow_up', 'arrow_down', 'arrow_left', 'arrow_right', 'w', 'a', 's', 'd')


def configure(offscreen=False):
    loadPrcFileData('', f'''
        window-title Autonomous Escape v5 - Docking Run
        win-size {W} {H}
        framebuffer-multisample 1
        multisamples 4
        sync-video 1
        textures-power-2 none
        show-frame-rate-meter 0
        {'window-type offscreen' if offscreen else ''}
        {'audio-library-name null' if offscreen else ''}
    ''')


def heading_of(d):
    """Panda heading (degrees) for a grid direction (dr, dc)."""
    return math.degrees(math.atan2(-d[1], -d[0]))


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def angle_lerp(a, b, t):
    return a + ((b - a + 180) % 360 - 180) * t


class Game(Effects):
    MODES = MODES

    def __init__(self, base, mode='rover', rover='A*', alien='Core', seed=None, speed=2, camera=None,
                 screen='title'):
        from ui3d import meshes, world
        from ui3d.audio import Sound
        self.base = base
        self.meshes, self.W3 = meshes, world
        self.ui = UI(base)
        self.sound = Sound(base)
        self.glow = meshes.glow_texture()
        self.mode = mode
        self.rover_name, self.alien_name = rover, alien
        self.seed = seed if seed is not None else random.randrange(1000)
        self.speed = speed
        self.camera_mode = camera
        self.abilities = True
        self.progress = story.load_progress()
        self.mission = None
        self.screen = None
        self.phase = None
        self.ui_root = None
        self.overlay = None
        self.hud = None
        self.state = None
        self.world = None
        self.world_cache = {}
        self.game_nodes = []
        self.menu_nodes = []
        self.show_plan = True
        self.show_cut = False
        self.paused = False
        self.waypoint = None
        self.drag = None
        self.look = [0.0, 0.0]          # pilot free-look yaw/pitch
        self.orbit = [0.0, 2.3]         # chase yaw offset, distance
        self.over = [0.0, 1.0]          # overhead yaw, zoom
        self.fov_zoom = 1.0
        base.setBackgroundColor(0.01, 0.01, 0.02, 1)
        base.disableMouse()
        base.camLens.setNearFar(0.03, 2000)
        self.sky = world.build_sky(base, self.glow)
        world.dress_sky(self.sky, self.glow)
        from ui3d import spacelook
        self.filters = spacelook.bloom(base)
        self.sun = world.lights(base, base.render, DEFAULT.size)
        self.weapon = 'boulder'
        self.setup_shadows()
        self.keys()
        base.taskMgr.add(self.tick, 'tick')
        self.backdrop_seed = story.MISSIONS[0].seed
        if screen == 'title':
            self.show_title()

    # ------------------------------------------------------------ helpers
    def now(self):
        return self.base.taskMgr.globalClock.getFrameTime()

    def world_xy(self, p):
        return float(p[1]), float(-p[0])

    def gz(self, x, y):
        return self.ground.z(x, y)

    def setup_shadows(self):
        try:
            from panda3d.core import CullFaceAttrib, RenderState
            self.base.render.setShaderAuto()     # space: no ground to catch shadows, skip shadow mapping
        except Exception:
            pass

    def card(self, parent, size, color, billboard=True):
        cm = CardMaker('glow')
        cm.setFrame(-size / 2, size / 2, -size / 2, size / 2)
        np_ = parent.attachNewNode(cm.generate())
        np_.setTexture(self.glow)
        np_.setColor(*color)
        if billboard:
            np_.setBillboardPointEye()
        self.meshes.additive(np_)
        return np_

    def use_world(self, seed, gmap):
        if self.world is not None:
            self.world.detachNode()
        if seed not in self.world_cache:
            if len(self.world_cache) > 3:
                for k in [k for k in self.world_cache if k != self.backdrop_seed]:
                    self.world_cache.pop(k)[0].removeNode()
            self.world_cache[seed] = self.W3.build_world(self.base.render, gmap, self.glow)
        self.world, self.ground, self.fields = self.world_cache[seed]
        self.world.reparentTo(self.base.render)

    def keys(self):
        b = self.base
        for k in DRIVE_KEYS:
            b.accept(k, self.on_key, [k])
            b.accept(k + '-repeat', self.on_key, [k, True])
        for k in ('enter', 'space', 'escape', 'r', 'n', 'v', 'p', 'c', 'q', 'e', '+', '=', '-', 'm', 'tab', 'h',
                  'f1', '1', '2'):
            b.accept(k, self.on_key, [k])
        b.accept('mouse1', self.on_click, [1])
        b.accept('mouse3', self.on_click, [3])
        b.accept('mouse3-up', self.end_drag)
        b.accept('wheel_up', self.on_wheel, [-1])
        b.accept('wheel_down', self.on_wheel, [1])

    # ------------------------------------------------------------ screens
    def clear_screen(self):
        if self.ui_root is not None:
            self.ui_root.removeNode()
            self.ui_root = None
        self.close_overlay()

    def close_overlay(self):
        if self.overlay is not None:
            self.overlay.removeNode()
            self.overlay = None

    def leave_game(self):
        if self.hud:
            self.hud.destroy()
            self.hud = None
        for n in self.game_nodes:
            n.removeNode()
        self.game_nodes = []
        self.state = None
        self.phase = None
        self.paused = False
        self.waypoint = None
        self.sound.motor(False)

    def backdrop(self):
        """The title/menu scenery: mission 1's field with the shuttle and saucer."""
        if self.state is not None:
            self.leave_game()
        if self.menu_nodes:
            return
        from escape import mapgen
        gmap = mapgen.generate(self.backdrop_seed)
        self.use_world(self.backdrop_seed, gmap)
        root = self.base.render.attachNewNode('menu_dyn')
        self.menu_nodes.append(root)
        station = self.W3.station_model(self.glow)
        x, y = self.world_xy(gmap.goal)
        station.setPos(x, y, 0)
        station.reparentTo(root)
        self.menu_station = station
        sau = models.SaucerModel(self.glow)
        ufo = sau.root
        ufo.reparentTo(root)
        self.menu_saucer = sau
        self.menu_ufo = ufo
        self.menu_map = gmap

    def drop_backdrop(self):
        for n in self.menu_nodes:
            n.removeNode()
        self.menu_nodes = []

    def menu_screen(self, name, items, sel=0):
        self.backdrop()
        self.clear_screen()
        self.screen = name
        self.items = items
        self.sel = sel
        self.ui_root = self.base.aspect2d.attachNewNode(name)

    def show_title(self):
        items = [('CAMPAIGN', self.show_missions, True), ('FREE PLAY', self.show_setup, True),
                 ('HOW TO PLAY', self.show_help, True), ('QUIT', self.quit, True)]
        self.menu_screen('title', items, getattr(self, 'title_sel', 0))
        self.draw_title()

    def draw_title(self):
        root = self.ui_root
        root.getChildren().detach()
        ui = self.ui
        a = self.base.getAspectRatio()
        ui.text(root, 'AUTONOMOUS ESCAPE', (-a + 0.14, 0.62), 0.14, TEXT, bold=True)
        ui.text(root, 'DOCKING RUN  ·  VERSION 5  ·  DEEP SPACE', (-a + 0.15, 0.52), 0.05, CYAN)
        ui.text(root, 'A shuttle, a station, and an alien that wants you lost in the debris.', (-a + 0.15, 0.44), 0.042,
                MUTED)
        for i, (label, fn, ok) in enumerate(self.items):
            ui.button(root, label, -a + 0.5, 0.2 - i * 0.15, 0.72, 0.11, self.pick_item, [i],
                      selected=i == self.sel, enabled=ok, scale=0.05)
        done = sum(self.progress.values())
        ui.text(root, f'Campaign stars: {done} / {3 * len(story.MISSIONS)}', (-a + 0.15, -0.46), 0.04, YELLOW)
        ui.text(root, 'Arrows + Enter, or click  ·  M mutes sound', (-a + 0.15, -0.9), 0.036, MUTED)

    def pick_item(self, i):
        self.sel = i
        self.sound.play('click')
        label, fn, ok = self.items[i]
        if ok:
            if self.screen == 'title':
                self.title_sel = i
            fn()

    def show_missions(self):
        items = []
        for i, m in enumerate(story.MISSIONS):
            items.append((m.title, (lambda i=i: self.show_briefing(i)), story.unlocked(i, self.progress)))
        items.append(('BACK', self.show_title, True))
        first = next((i for i, m in enumerate(story.MISSIONS)
                      if story.unlocked(i, self.progress) and not self.progress.get(m.key)), 0)
        self.menu_screen('missions', items, getattr(self, 'mission_sel', first))
        self.draw_missions()

    def draw_missions(self):
        root = self.ui_root
        root.getChildren().detach()
        ui = self.ui
        a = self.base.getAspectRatio()
        ui.panel(root, -a + 0.08, a - 0.08, 0.5, 0.92, color=(0.03, 0.03, 0.09, 0.82))
        ui.text(root, 'CAMPAIGN', (-a + 0.14, 0.83), 0.06, CYAN, bold=True)
        for k, line in enumerate(story.PROLOGUE):
            ui.text(root, line, (-a + 0.14, 0.76 - k * 0.05), 0.038, TEXT if k else YELLOW)
        for i, m in enumerate(story.MISSIONS):
            y = 0.36 - i * 0.16
            ok = story.unlocked(i, self.progress)
            sel = i == self.sel
            ui.button(root, '', -0.25, y, 1.9, 0.13, self.pick_item, [i], selected=sel, enabled=ok)
            ui.text(root, f'{i + 1}', (-1.12, y - 0.02), 0.06, CYAN if ok else DIM, TextNode.ACenter, bold=True)
            ui.text(root, m.title if ok else 'LOCKED', (-1.0, y + 0.005), 0.05, TEXT if ok else DIM, bold=True)
            if ok:
                sub = f'{m.place}  ·  alien: {m.alien}  ·  {m.budget} strikes' + (f'  ·  new: {m.unlock}'
                                                                                  if m.unlock else '')
                ui.text(root, sub, (-1.0, y - 0.045), 0.032, MUTED)
            got = self.progress.get(m.key, 0)
            for s_ in range(3):
                star(root, 0.42 + s_ * 0.075, y, 0.03, YELLOW if s_ < got else (0.25, 0.24, 0.32, 1))
        ui.button(root, 'BACK', -0.25, 0.36 - 5 * 0.16, 0.5, 0.09, self.pick_item, [len(story.MISSIONS)],
                  selected=self.sel == len(story.MISSIONS))
        ui.panel(root, -0.95, 0.45, -0.62, -0.53, color=(0.03, 0.03, 0.09, 0.75), edge=None)
        ui.text(root, 'Win a mission to unlock the next  ·  more fuel left = more stars', (-0.25, -0.59), 0.034,
                TEXT, TextNode.ACenter)

    def show_briefing(self, i):
        self.mission_sel = i
        m = story.MISSIONS[i]
        items = [('DEPLOY', lambda: self.start_mission(i), True), ('BACK', self.show_missions, True)]
        self.menu_screen('briefing', items)
        self.brief_t0 = self.now()
        self.brief_mission = m
        self.draw_briefing()

    def draw_briefing(self):
        root = self.ui_root
        root.getChildren().detach()
        ui = self.ui
        m = self.brief_mission
        i = story.MISSIONS.index(m)
        ui.panel(root, -1.25, 1.25, -0.72, 0.78, color=(0.03, 0.03, 0.09, 0.88), edge=CYAN)
        ui.text(root, f'MISSION {i + 1}  ·  {m.place.upper()}', (-1.17, 0.66), 0.045, CYAN)
        ui.text(root, m.title.upper(), (-1.17, 0.54), 0.1, TEXT, bold=True)
        self.brief_lines = [ui.text(root, '', (-1.17, 0.38 - k * 0.075), 0.044, TEXT) for k in range(len(m.briefing))]
        r = m.rules()
        facts = [f'Alien AI: {m.alien}', f'Strikes: {m.budget}, every {r.alien_every} turns',
                 f'Fuel: {m.fuel_mult}x the shortest route', 'Radar pulse: ' + (f'{m.scan_cost} fuel' if m.scan_cost
                                                                              else 'offline'),
                 f'Shield: {m.shields} hit' + ('s' if m.shields != 1 else '') if m.shields else 'Shield: offline',
                 f'Stars: escape / {m.stars[0]}+ / {m.stars[1]}+ score']
        for k, f in enumerate(facts):
            ui.text(root, f, (0.42, 0.38 - k * 0.065), 0.036, MUTED)
        for k, (label, fn, ok) in enumerate(self.items):
            ui.button(root, label, -0.82 + k * 0.62, -0.58, 0.5, 0.11, self.pick_item, [k], selected=k == self.sel)

    def show_setup(self):
        items = [('LAUNCH', self.start_free, True), ('BACK', self.show_title, True)]
        self.menu_screen('setup', items)
        self.setup_row = 0
        self.draw_setup()

    def setup_rows(self):
        return [('Mode', dict(MODES)[self.mode], True),
                ('Rover AI', self.rover_name if self.mode != 'rover' else 'You', self.mode != 'rover'),
                ('Alien AI', self.alien_name if self.mode != 'alien' else 'You', self.mode != 'alien'),
                ('Map', f'#{self.seed}', True),
                ('Speed', f'{self.speed} moves / s', self.mode != 'rover'),
                ('Camera', self.default_camera(), True),
                ('Pulse + shield', 'On' if self.abilities else 'Off (benchmark rules)', True)]

    def default_camera(self):
        return self.camera_mode or {'rover': 'Pilot', 'ai': 'Chase', 'alien': 'Overhead'}[self.mode]

    def setup_change(self, row, d):
        self.sound.play('click')
        if row == 0:
            keys = [m for m, _ in MODES]
            self.mode = keys[(keys.index(self.mode) + d) % len(keys)]
            self.camera_mode = None
        elif row == 1 and self.mode != 'rover':
            names = list(ROVER_BY_NAME)
            self.rover_name = names[(names.index(self.rover_name) + d) % len(names)]
        elif row == 2 and self.mode != 'alien':
            names = list(ALIEN_BY_NAME)
            self.alien_name = names[(names.index(self.alien_name) + d) % len(names)]
        elif row == 3:
            self.seed = (self.seed + d) % 1000
        elif row == 4:
            i = SPEEDS.index(self.speed) if self.speed in SPEEDS else 1
            self.speed = SPEEDS[(i + d) % len(SPEEDS)]
        elif row == 5:
            self.camera_mode = CAMERAS[(CAMERAS.index(self.default_camera()) + d) % len(CAMERAS)]
        elif row == 6:
            self.abilities = not self.abilities
        self.setup_row = row
        self.draw_setup()

    def draw_setup(self):
        from direct.gui.DirectGui import DirectButton, DGG
        root = self.ui_root
        root.getChildren().detach()
        ui = self.ui
        a = self.base.getAspectRatio()
        x0 = -a + 0.14
        ui.text(root, 'FREE PLAY', (x0, 0.78), 0.08, TEXT, bold=True)
        ui.text(root, 'Any rover against any alien, any map. Pulse + shield off = the benchmark rules.',
                (x0, 0.7), 0.038, MUTED)
        for i, (label, value, ok) in enumerate(self.setup_rows()):
            y = 0.52 - i * 0.12
            sel = i == self.setup_row and self.sel == -1
            ui.panel(root, x0, x0 + 1.3, y - 0.05, y + 0.05, edge=CYAN if sel else None)
            ui.text(root, label, (x0 + 0.05, y - 0.016), 0.045, TEXT if ok else DIM)
            ui.text(root, value, (x0 + 0.93, y - 0.016), 0.042, TEXT if ok else DIM, TextNode.ACenter)
            if ok:
                for dx, ch, d in ((0.62, '<', -1), (1.24, '>', 1)):
                    DirectButton(parent=root, text=ch, text_font=ui.bold, scale=0.05, pos=(x0 + dx, 0, y - 0.016),
                                 text_fg=CYAN, frameColor=(0.15, 0.14, 0.32, 1), pad=(0.3, 0.1), relief=DGG.FLAT,
                                 command=self.setup_change, extraArgs=[i, d])
        for k, (label, fn, ok) in enumerate(self.items):
            ui.button(root, label, x0 + 0.33 + k * 0.66, 0.52 - 7 * 0.12 - 0.04, 0.6, 0.1, self.pick_item, [k],
                      selected=self.sel == k)

    def show_help(self, back=None):
        self.help_back = back or self.show_title
        items = [('BACK', self.help_back, True)]
        self.menu_screen('help', items) if back is None else None
        root = self.ui_root if back is None else self.make_overlay()
        ui = self.ui
        ui.panel(root, -1.3, 1.3, -0.85, 0.85, color=(0.03, 0.03, 0.09, 0.92), edge=CYAN)
        ui.text(root, 'HOW TO PLAY', (-1.2, 0.72), 0.07, TEXT, bold=True)
        cols = [
            ('DRIVE (pilot / chase cam)', ['W / Up: forward one tile', 'S / Down: reverse one tile',
                                           'A D / Left Right: turn 90° (free)', 'Space: wait a turn',
                                           'Drive into a warm boulder: laser it (2 fuel)',
                                           'Q: radar pulse (maps 10 tiles)', 'Click the ground: autopilot']),
            ('CAMERA', ['V: Pilot / Chase / Overhead', 'Right-drag: look around / orbit', 'Wheel: zoom',
                        'Tab: orbital map', 'P route  ·  C alien cut']),
            ('THE RULES', ['Every move or wait costs 1 fuel', 'The saucer strikes every 3 turns',
                           'Boulders block; warm ones can be lasered', 'Debris lands 2 turns after the red ring',
                           'It fuses boulders and hits for 6 fuel', 'Reach the pad to launch. Fuel left = score']),
        ]
        for k, (head, lines) in enumerate(cols):
            x = -1.2 + k * 0.85
            ui.text(root, head, (x, 0.56), 0.042, CYAN, bold=True)
            for j, line in enumerate(lines):
                ui.text(root, line, (x, 0.46 - j * 0.075), 0.034, TEXT, wrap=21)
        ui.text(root, 'Esc / Enter: back', (0, -0.76), 0.04, MUTED, TextNode.ACenter)

    def make_overlay(self):
        self.close_overlay()
        self.overlay = self.base.aspect2d.attachNewNode('overlay')
        return self.overlay

    def quit(self):
        sys.exit(0)

    # ------------------------------------------------------------ start a game
    def start_mission(self, i):
        self.mission = story.MISSIONS[i]
        m = self.mission
        self.start_game(seed=m.seed, mode='rover', alien=m.alien, rules=m.rules(), camera=self.camera_mode or 'Pilot')

    def start_free(self):
        self.mission = None
        rules = DEFAULT
        if self.abilities:
            from dataclasses import replace
            rules = replace(DEFAULT, scan_cost=2, shields=1)
        self.start_game(seed=self.seed, mode=self.mode, alien=self.alien_name, rules=rules,
                        camera=self.default_camera())

    def restart(self):
        if self.mission:
            self.start_mission(story.MISSIONS.index(self.mission))
        else:
            self.start_free()

    def start_game(self, seed=None, mode=None, alien=None, rules=None, camera=None, intro=True):
        W3 = self.W3
        self.clear_screen()
        self.leave_game()
        self.drop_backdrop()
        self.screen = 'game'
        if seed is not None:
            self.seed = seed
        self.mode = mode or self.mode
        alien = alien or self.alien_name
        self.cam = camera or self.default_camera()
        s = self.state = GameState.new(self.seed, rules or DEFAULT)
        self.rover_ai = make_rover(self.rover_name, self.seed) if self.mode != 'rover' else None
        self.alien_ai = make_alien(alien, self.seed, **({'time_limit': 0.5} if alien == 'MCTS' else {})) \
            if self.mode != 'alien' else None
        for ag in (self.rover_ai, self.alien_ai):
            if ag:
                ag.reset()
        self.core_view = CoreAlien()
        self.use_world(self.seed, s.map)
        dyn = self.dyn = self.base.render.attachNewNode('dynamic')
        self.game_nodes.append(dyn)
        self.rover_np = dyn.attachNewNode('rover_pose')     # the pilot camera rides on this
        self.rover = W3.ShuttleModel(self.glow)
        self.rover_body = self.rover.root
        self.rover_body.reparentTo(self.rover_np)
        self.shield_np = self.meshes.sphere('shield', (0.4, 0.9, 1, 0.12), sub=3)
        self.shield_np.setScale(0.55, 0.55, 0.42)
        self.shield_np.setZ(0.2)
        self.meshes.translucent(self.shield_np)
        self.shield_np.setLightOff(1)
        self.shield_np.setShaderOff(1)
        self.shield_np.reparentTo(self.rover_np)
        self.saucer = models.SaucerModel(self.glow)
        self.ufo = self.saucer.root
        self.ufo.reparentTo(dyn)
        self.ufo_pos = Vec3(s.map.size / 2, -s.map.size / 2, 5.0)
        self.ufo_vel = Vec3(0, 0, 0)
        self.ufo_goal, self.ufo_until = None, 0
        self.ufo_shadow = self.meshes.disc('ufoshadow', 0.8, (0, 0, 0, 0.6))
        self.meshes.translucent(self.ufo_shadow)
        self.ufo_shadow.setLightOff(1)
        self.ufo_shadow.setShaderOff(1)
        self.ufo_shadow.reparentTo(dyn)
        self.decals = dyn.attachNewNode('decals')
        self.styles, self.proj_style = {}, {}
        self.dust_t = 0.0
        self.tilt = [0.0, 0.0, 0.0]
        self.make_pad(s.map.goal)
        self.make_base_station(s.map.start)
        self.radar = self.meshes.additive(self.meshes.ring('radar', s.rules.radar + 0.42, s.rules.radar + 0.5, CYAN))
        self.radar.setAlphaScale(0.45)
        self.radar.reparentTo(dyn)
        self.route_np = dyn.attachNewNode('route')
        self.cut_np = dyn.attachNewNode('cut')
        self.hover_np = dyn.attachNewNode('hover')
        self.way_np = dyn.attachNewNode('waypoint')
        self.boulder_nodes, self.meteors, self.effects = {}, {}, []
        self.heading = self.initial_heading()
        self.motion = Motion(self.world_xy(s.rover), heading_of(self.heading))
        self.reverse_next = False
        self.last_tick = self.now()
        self.next_time = self.now() + 0.8
        self.cam_pos = None
        self.shake = 0
        self.look = [0.0, 0.0]
        self.fov_zoom = 1.0
        self.waypoint = None
        self.paused = False
        self.step_once = False
        self.dirty = True
        self.warned = set()
        self.start_dist = manhattan(s.rover, s.map.goal)
        self.overlays()
        from ui3d.hud import Hud
        self.hud = Hud(self)
        self.place_rover(1 / 60)
        if intro:
            self.begin_intro()
        else:
            self.phase = 'play'
            self.on_play()

    def initial_heading(self):
        s = self.state
        _, plan = astar(s.map, {}, s.rover, s.map.goal, s.rules)
        if len(plan) > 1:
            return (plan[1][0] - plan[0][0], plan[1][1] - plan[0][1])
        return (1, 0)

    def make_pad(self, p):
        """The goal: the docking port of an orbital station."""
        x, y = self.world_xy(p)
        root = self.dyn.attachNewNode('pad')
        root.setPos(x, y, 0)
        self.station = self.W3.station_model(self.glow)
        self.station.reparentTo(root)
        self.pad_glow = self.card(root, 2.4, CYAN[:3] + (0.35,))
        self.pad_glow.setZ(0.18)
        self.shuttle = root.attachNewNode('none')
        self.flame = root.attachNewNode('none')
        self.pad = root

    def make_base_station(self, p):
        x, y = self.world_xy(p)
        root = self.dyn.attachNewNode('base')
        root.setPos(x, y, 0)
        self.W3.launch_dock_model(self.glow).reparentTo(root)
        self.base_beacon = self.card(root, 0.35, (1, 0.25, 0.25, 1))
        self.base_beacon.setPos(-0.6, 0.6, 1.0)

    def begin_intro(self):
        self.phase = 'intro'
        self.phase_t0 = self.now()
        self.hud.hide()
        root = self.make_overlay()
        ui = self.ui
        if self.mission:
            i = story.MISSIONS.index(self.mission)
            ui.text(root, f'MISSION {i + 1}', (0, 0.18), 0.05, CYAN, TextNode.ACenter)
            ui.text(root, self.mission.title.upper(), (0, 0.05), 0.12, TEXT, TextNode.ACenter, bold=True)
            ui.text(root, self.mission.place, (0, -0.05), 0.045, MUTED, TextNode.ACenter)
        else:
            ui.text(root, dict(MODES)[self.mode].upper(), (0, 0.05), 0.1, TEXT, TextNode.ACenter, bold=True)
            ui.text(root, f'Map #{self.seed}', (0, -0.05), 0.045, MUTED, TextNode.ACenter)
        ui.text(root, 'Space: skip', (0, -0.85), 0.035, MUTED, TextNode.ACenter)
        self.sound.play('whoosh')

    def end_intro(self):
        self.close_overlay()
        self.hud.show()
        self.phase = 'play'
        self.cam_pos = None
        self.next_time = self.now() + 0.6
        self.on_play()

    def on_play(self):
        if self.mode == 'rover':
            self.hud.radio('start', dist=self.start_dist)

    def intro_camera(self, t):
        n = self.state.map.size
        rx, ry, rz = self.rover_np.getX(), self.rover_np.getY(), self.rover_np.getZ()
        h = self.rover_np.getH()
        fx, fy = -math.sin(math.radians(h)), math.cos(math.radians(h))
        start = Vec3(n * 0.9, -n * 1.4, n * 0.9)
        end = Vec3(rx - fx * 2.3, ry - fy * 2.3, rz + 1.4)
        f = ease(t)
        pos = start + (end - start) * f
        look_a = Vec3(n / 2, -n / 2, 0)
        look_b = Vec3(rx + fx * 1.4, ry + fy * 1.4, rz + 0.1)
        cam = self.base.camera
        cam.reparentTo(self.base.render)
        cam.setPos(pos)
        cam.lookAt(look_a + (look_b - look_a) * f)
        self.base.camLens.setFov(lerp(50, 68, f))
        self.rover_body.show()

    def begin_outro(self):
        s = self.state
        self.phase = 'outro'
        self.phase_t0 = self.now()
        self.waypoint = None
        self.sound.motor(False)
        self.hud.hide()
        if s.result == ROVER:
            self.hud.radio('win', once=False)
            self.sound.play('launch')
            self.boarded = False
            self.dock_from = self.rover_np.getPos()
        else:
            self.hud.radio('lose', once=False)
            self.sound.play('lose')
            root = self.make_overlay()
            from direct.gui.OnscreenImage import OnscreenImage
            from .widgets import DynamicTexture
            self.static_tex = DynamicTexture(160, 100, 'static')
            img = OnscreenImage(image=self.static_tex.tex, parent=self.base.render2d, scale=(1, 1, 1))
            img.setTransparency(TransparencyAttrib.MAlpha)
            img.reparentTo(root)
            self.static_img = img
            self.ui.text(root, 'SIGNAL LOST', (0, 0.05), 0.12, RED, TextNode.ACenter, bold=True)
            self.ui.text(root, s.reason, (0, -0.07), 0.05, TEXT, TextNode.ACenter)

    def outro_frame(self, now, final=False):
        """Shuttle launch or signal-lost sequence; `final` keeps the camera
        moving behind the debrief without advancing the phase."""
        s = self.state
        t = now - self.phase_t0
        cam = self.base.camera
        cam.reparentTo(self.base.render)
        self.base.camLens.setFov(60)
        if s.result == ROVER:
            px, py = self.pad.getX(), self.pad.getY()
            port = Vec3(px + 0.35, py - 0.35, 0.18)
            u = min(1.0, t / 2.5)
            e = u * u * (3 - 2 * u)
            self.rover_np.setPos(self.dock_from + (port - self.dock_from) * e)
            self.rover_np.setH(-45 + 180 if False else self.rover_np.getH() + ((-135) - self.rover_np.getH()) * 0.05)
            guides = self.station.getPythonTag('guides')
            guides.setAlphaScale(1.0 if int(now * 6) % 2 else 0.3)
            if t > 2.5 and not self.boarded:
                self.boarded = True
                self.sound.play('win')
                self.hud.flash('DOCKED')
            a = 0.25 * t
            cam.setPos(px + 4 * math.cos(a) - 1, py - 4 * math.sin(a) - 2, 2.2)
            cam.lookAt(px + 0.5, py - 0.5, 0.4)
            if t > 5.5 and not final:
                self.show_debrief()
        elif self.phase == 'outro':
            import pygame as pg
            surf = pg.Surface((160, 100), pg.SRCALPHA)
            arr = np.random.randint(0, 255, (160, 100), np.uint8)
            alpha = int(min(200, 60 + t * 60))
            rgb = np.stack([arr, arr, arr], -1)
            pg.surfarray.blit_array(surf, rgb)
            surf.set_alpha(None)
            surf2 = pg.Surface((160, 100), pg.SRCALPHA)
            surf2.blit(surf, (0, 0))
            surf2.fill((255, 255, 255, alpha), special_flags=pg.BLEND_RGBA_MULT)
            self.static_tex.load(surf2)
            rx, ry, rz = self.rover_np.getX(), self.rover_np.getY(), self.rover_np.getZ()
            cam.setPos(rx + 1.5, ry - 2.5 - t * 0.6, rz + 1.2 + t * 0.9)
            cam.lookAt(rx, ry, rz)
            if t > 3.2 and not final:
                self.show_debrief()

    def show_debrief(self):
        s = self.state
        self.phase = 'debrief'
        won = s.result == ROVER
        root = self.make_overlay()
        ui = self.ui
        stars_got = 0
        if self.mission:
            stars_got = story.stars_for(self.mission, won, s.score)
            if stars_got > self.progress.get(self.mission.key, 0):
                self.progress[self.mission.key] = stars_got
                story.save_progress(self.progress)
        ui.panel(root, -0.95, 0.95, -0.62, 0.6, color=(0.03, 0.03, 0.09, 0.93), edge=CYAN if won else ORANGE)
        title = ('MISSION COMPLETE' if won else 'MISSION FAILED') if self.mission else \
            ('ROVER ESCAPED' if won else 'ROVER TRAPPED')
        ui.text(root, title, (0, 0.44), 0.1, CYAN if won else ORANGE, TextNode.ACenter, bold=True)
        ui.text(root, s.reason.replace('Reached the launch pad', 'Docked with Ares Gate'), (0, 0.34), 0.048, TEXT,
                TextNode.ACenter)
        if self.mission:
            for k in range(3):
                star(root, -0.15 + k * 0.15, 0.2, 0.06, YELLOW if k < stars_got else (0.25, 0.24, 0.32, 1))
        stats = [('Escape score', f'{s.score:.0f}'), ('Fuel left', f'{max(0, s.fuel)} / {s.fuel0}'),
                 ('Turns', f'{s.tick}'), ('Laser shots', f'{s.shots}'), ('Debris hits', f'{s.hits}'),
                 ('Radar pulses', f'{s.scans}')]
        for k, (a, b) in enumerate(stats):
            x = -0.62 + (k % 3) * 0.62
            y = 0.04 - (k // 3) * 0.13
            ui.text(root, a, (x, y), 0.034, MUTED, TextNode.ACenter)
            ui.text(root, b, (x, y - 0.06), 0.05, TEXT, TextNode.ACenter, bold=True)
        last = self.mission is story.MISSIONS[-1]
        if self.mission and won and last:
            for k, line in enumerate(story.EPILOGUE):
                ui.text(root, line, (0, -0.27 - k * 0.045), 0.032, YELLOW if k == 2 else TEXT, TextNode.ACenter)
        items = []
        if self.mission:
            i = story.MISSIONS.index(self.mission)
            if won and not last:
                items.append(('NEXT MISSION', lambda: self.show_briefing(i + 1)))
            items.append(('RETRY', self.restart))
            items.append(('MISSIONS', self.show_missions))
        else:
            items += [('REPLAY', self.restart), ('NEW MAP', self.new_map), ('MENU', self.show_title)]
        self.debrief_items = items
        for k, (label, fn) in enumerate(items):
            x = (k - (len(items) - 1) / 2) * 0.55
            ui.button(root, label, x, -0.52, 0.48, 0.1, fn, selected=k == 0, scale=0.042)

    def new_map(self):
        self.seed = random.randrange(1000)
        self.start_free()

    # ------------------------------------------------------------ pause
    def toggle_pause(self):
        if self.phase != 'play':
            return
        if self.overlay is not None:
            self.close_overlay()
            self.paused = False
            return
        self.paused = True
        root = self.make_overlay()
        ui = self.ui
        ui.panel(root, -0.5, 0.5, -0.45, 0.45, color=(0.03, 0.03, 0.09, 0.92), edge=CYAN)
        ui.text(root, 'PAUSED', (0, 0.3), 0.08, TEXT, TextNode.ACenter, bold=True)
        items = [('RESUME', self.toggle_pause), ('RESTART', self.restart),
                 ('CONTROLS', lambda: self.show_help(back=self.toggle_pause)),
                 ('QUIT TO MENU', self.show_missions if self.mission else self.show_title)]
        for k, (label, fn) in enumerate(items):
            ui.button(root, label, 0, 0.15 - k * 0.13, 0.6, 0.1, fn, scale=0.045)

    # ------------------------------------------------------------ turns
    def interval(self):
        return 1.0 / self.speed

    def overlays(self):
        s = self.state
        v = s.rover_view()
        known, self.plan = astar(s.map, v.belief, s.rover, s.map.goal, s.rules)
        self.cut = self.core_view.plan(s)[0] if self.mode != 'rover' else []
        self.costs = (known, s.route_cost())

    def do_rover(self, action):
        s = self.state
        landing = [p for p, t in s.pending.items() if t == 1]
        hits, shields = s.hits, s.shields
        frm = s.rover
        dying = None
        if action[0] == 'shoot':
            node = self.boulder_nodes.pop(action[1], None)
            dying = node[0] if node else None
        s.apply_rover(action)
        now = self.now()
        self.last_tick = now
        if action[0] == 'move' and s.rover != frm:
            d = (s.rover[0] - frm[0], s.rover[1] - frm[1])
            reverse = self.reverse_next
            self.reverse_next = False
            if not reverse:
                self.heading = d
            self.motion.push_move(self.world_xy(s.rover), reverse=reverse)
        elif action[0] == 'shoot':
            p = action[1]
            d = (p[0] - s.rover[0], p[1] - s.rover[1])
            self.heading = d
            self.motion.push_turn(heading_of(d))
            self.fire_laser(p, dying, self.styles.get(p, 'rock'))
            self.sound.play('laser')
            if self.mode == 'rover':
                self.hud.radio('laser')
        elif action[0] == 'scan':
            self.add_scan()
            self.sound.play('scan')
            if self.mode == 'rover':
                found = int(s.log[-1][1].split()[3]) if s.log else 0
                self.hud.radio('scan', once=False, found=found)
        for p in landing:
            self.plunge(p, self.proj_style.pop(p, 'rock'))
        if s.hits > hits:
            self.shake = 0.9
            self.hud.hurt()
            self.hud.flash(f'Debris hit! -{s.rules.debris_damage} fuel')
            if self.mode == 'rover':
                self.hud.radio('hit', once=False, dmg=s.rules.debris_damage)
        if s.shields < shields:
            self.add_shield_flash()
            self.sound.play('shield')
            if self.mode == 'rover':
                self.hud.radio('shield', once=False)
        self.dirty = True
        if s.turn == ALIEN and self.mode == 'rover':
            self.next_time = now + 0.5

    def do_alien(self, action):
        s = self.state
        before = dict(s.blk)
        s.apply_alien(action)
        now = self.now()
        if action:
            kind, p = action
            visible = self.mode != 'rover' or s.in_radar(p)
            if self.mode == 'rover':
                self.hud.radio('first_strike')
            if kind == 'boulder':
                style = random.choice(('rock', 'rock', 'wreck', 'crystal'))
                self.styles[p] = style
                if visible:
                    x, y = self.world_xy(p)
                    self.ufo_goal = Vec3(x, y, self.gz(x, y) + 2.8)
                    self.ufo_until = now + 1.9
                    self.drop_obstacle(p, style)
                    if self.mode == 'rover':
                        self.hud.radio('drop_seen')
            else:
                style = random.choice(('rock', 'crystal', 'wreck'))
                self.proj_style[p] = style
                if not before.get(p):
                    self.styles[p] = style
                self.ufo_goal = Vec3(self.ufo_pos.x, self.ufo_pos.y, self.ufo_pos.z + 0.6)
                self.ufo_until = now + 1.3
                self.launch_debris(style)
                if self.mode == 'rover':
                    self.hud.radio('debris')
        self.dirty = True
        self.next_time = now + self.interval()

    def busy(self):
        """The rover may take its next action once the drive ahead is short,
        so held keys and AI moves chain into continuous driving."""
        return self.motion.remaining() > 0.45

    def can_act(self):
        s = self.state
        return (self.phase == 'play' and not self.paused and s and not s.result and s.turn == ROVER
                and not self.busy())

    def drive(self, key):
        if not self.can_act():
            return
        self.waypoint = None
        absolute = {'arrow_up': (-1, 0), 'w': (-1, 0), 'arrow_down': (1, 0), 's': (1, 0),
                    'arrow_left': (0, -1), 'a': (0, -1), 'arrow_right': (0, 1), 'd': (0, 1)}
        if self.cam == 'Overhead':
            self.heading = absolute[key]
            self.try_move(self.heading)
            return
        dr, dc = self.heading
        if key in ('arrow_left', 'a', 'arrow_right', 'd'):
            self.heading = (-dc, dr) if key in ('arrow_left', 'a') else (dc, -dr)
            self.motion.push_turn(heading_of(self.heading))
            self.hud.map_dirty = True
            return
        if key in ('arrow_up', 'w'):
            self.try_move((dr, dc))
        else:
            self.reverse_next = True
            if not self.try_move((-dr, -dc)):
                self.reverse_next = False

    def try_move(self, d):
        s = self.state
        tgt = (s.rover[0] + d[0], s.rover[1] + d[1])
        if not s.map.is_open(tgt):
            self.hud.flash('Asteroid cluster: no way through')
            return False
        b = s.blk.get(tgt, OPEN)
        if b == HARD:
            self.hud.flash('Fused rock: the laser cannot cut it')
            if self.mode == 'rover':
                self.hud.radio('fused')
            return False
        self.do_rover(('shoot', tgt) if b == SOFT else ('move', tgt))
        return True

    def autopilot(self):
        if not self.waypoint or not self.can_act():
            return
        s = self.state
        if s.rover == self.waypoint:
            self.waypoint = None
            self.hud.flash('Autopilot: arrived')
            self.hud.map_dirty = True
            return
        cost, path = astar(s.map, s.belief, s.rover, self.waypoint, s.rules)
        if cost == INF or len(path) < 2:
            self.waypoint = None
            self.hud.flash('Autopilot: no known way there')
            return
        nxt = path[1]
        if s.pending.get(nxt) == 1:
            self.waypoint = None
            self.hud.flash('Autopilot halted: debris landing ahead')
            self.sound.play('alarm')
            return
        self.heading = (nxt[0] - s.rover[0], nxt[1] - s.rover[1])
        if not self.try_move(self.heading):
            self.waypoint = None

    # ------------------------------------------------------------ input
    def on_key(self, k, repeat=False):
        if self.screen in ('title', 'missions', 'briefing', 'help'):
            if repeat:
                return
            if k in ('arrow_up', 'w', 'arrow_left', 'a'):
                self.sel = (self.sel - 1) % len(self.items)
            elif k in ('arrow_down', 's', 'arrow_right', 'd', 'tab'):
                self.sel = (self.sel + 1) % len(self.items)
            elif k in ('enter', 'space'):
                self.pick_item(self.sel)
                return
            elif k == 'escape':
                if self.screen == 'title':
                    self.quit()
                elif self.screen == 'briefing':
                    self.show_missions()
                else:
                    self.show_title()
                return
            elif k == 'm':
                self.sound.toggle_mute()
                return
            else:
                return
            self.sound.play('click')
            {'title': self.draw_title, 'missions': self.draw_missions, 'briefing': self.draw_briefing,
             'help': lambda: None}[self.screen]()
            if self.screen == 'missions':
                self.mission_sel = self.sel
            return
        if self.screen == 'setup':
            if repeat:
                return
            rows = len(self.setup_rows())
            if k in ('arrow_up', 'w'):
                self.setup_row = (self.setup_row - 1) % (rows + 1)
            elif k in ('arrow_down', 's'):
                self.setup_row = (self.setup_row + 1) % (rows + 1)
            elif k in ('arrow_left', 'a', 'arrow_right', 'd') and self.setup_row < rows:
                self.setup_change(self.setup_row, -1 if k in ('arrow_left', 'a') else 1)
                return
            elif k in ('enter', 'space'):
                self.start_free()
                return
            elif k == 'escape':
                self.show_title()
                return
            self.sel = -1 if self.setup_row < rows else 0
            self.draw_setup()
            return
        if self.screen != 'game':
            return
        s = self.state
        if k == 'm':
            muted = self.sound.toggle_mute()
            self.hud.flash('Sound off' if muted else 'Sound on')
            return
        if self.phase == 'intro':
            if k in ('space', 'enter', 'escape'):
                self.end_intro()
            return
        if self.phase == 'outro':
            if k in ('space', 'enter'):
                self.show_debrief()
            return
        if self.phase == 'debrief':
            if k in ('enter', 'space'):
                self.debrief_items[0][1]()
            elif k == 'escape':
                self.show_missions() if self.mission else self.show_title()
            elif k == 'r':
                self.restart()
            return
        if self.overlay is not None:              # pause menu or help
            if k in ('escape', 'enter', 'space', 'h', 'f1'):
                self.close_overlay()
                self.paused = False
            return
        if k == 'escape':
            self.toggle_pause()
        elif k in ('h', 'f1'):
            self.paused = True
            self.show_help(back=self.toggle_pause)
        elif k == 'r':
            self.restart()
        elif k == 'n' and not self.mission:
            self.new_map()
        elif k == 'v':
            self.cam = CAMERAS[(CAMERAS.index(self.cam) + 1) % len(CAMERAS)]
            self.cam_pos = None
            self.look = [0.0, 0.0]
        elif k == 'p':
            self.show_plan = not self.show_plan
            self.dirty = True
        elif k == 'c':
            self.show_cut = not self.show_cut
            self.dirty = True
        elif k == 'tab':
            m = self.hud.map_root
            m.hide() if not m.isHidden() else m.show()
        elif k in ('+', '='):
            self.speed = SPEEDS[min(len(SPEEDS) - 1, SPEEDS.index(self.speed) + 1)] if self.speed in SPEEDS else 2
        elif k == '-':
            self.speed = SPEEDS[max(0, SPEEDS.index(self.speed) - 1)] if self.speed in SPEEDS else 2
        elif self.mode == 'rover':
            if k == 'space' and self.can_act():
                self.waypoint = None
                self.do_rover(('wait', None))
            elif k == 'q' and self.can_act():
                if ('scan', None) in s.rover_actions():
                    self.do_rover(('scan', None))
                else:
                    self.hud.flash('Radar pulse offline' if not s.rules.scan_cost else 'Not enough fuel')
            elif k in DRIVE_KEYS:
                self.drive(k)
        elif k == 'space':
            if self.mode == 'alien' and s.turn == ALIEN and not s.result:
                self.do_alien(None)
            else:
                self.paused = not self.paused
        elif k == 'arrow_right' and self.mode == 'ai' and not repeat:
            self.paused = True
            self.step_once = True
        elif k in ('1', '2') and self.mode == 'alien':
            self.weapon = 'boulder' if k == '1' else 'debris'
            self.hud.flash('Weapon: ' + ('boulder drop' if k == '1' else 'meteor'))
        elif k in ('q', 'e') and self.cam == 'Overhead':
            self.over[0] += 15 if k == 'q' else -15

    def mouse(self):
        mw = self.base.mouseWatcherNode
        if mw and mw.hasMouse():
            m = mw.getMouse()
            return m.x, m.y
        return None

    def pick(self):
        """Grid tile under the mouse (ray against the ground plane)."""
        mw = self.base.mouseWatcherNode
        if not mw or not mw.hasMouse():
            return None
        near, far = Point3(), Point3()
        self.base.camLens.extrude(mw.getMouse(), near, far)
        render = self.base.render
        a = render.getRelativePoint(self.base.cam, near)
        b = render.getRelativePoint(self.base.cam, far)
        if a.z - b.z < 1e-6:
            return None
        t = (a.z - 0.05) / (a.z - b.z)
        p = a + (b - a) * t
        r, c = int(round(-p.y)), int(round(p.x))
        n = self.state.map.size
        return (r, c) if 0 <= r < n and 0 <= c < n else None

    def on_click(self, button):
        if self.screen != 'game' or self.phase != 'play' or self.overlay is not None:
            return
        s = self.state
        if self.mode == 'alien':
            p = self.pick()
            if p is None or s.result or s.turn != ALIEN:
                if button == 3:
                    self.drag = self.mouse()
                return
            a = self.alien_choice(p, button)
            if s.legal_alien(a):
                self.do_alien(a)
            else:
                self.hud.flash(self.why_illegal(p))
            return
        if button == 3:
            self.drag = self.mouse()
            self.drag_ref = (list(self.look), list(self.orbit), list(self.over))
            return
        if self.mode == 'rover':
            p = self.pick()
            if p is None or p == s.rover:
                return
            if not s.map.is_open(p) or s.belief.get(p) == HARD:
                self.hud.flash('Autopilot: pick open ground')
                return
            self.waypoint = p
            self.hud.flash(f'Autopilot engaged to {p}')
            self.hud.map_dirty = True
            self.sound.play('click')

    def end_drag(self):
        self.drag = None

    def on_wheel(self, d):
        if self.screen != 'game':
            return
        if self.cam == 'Pilot':
            self.fov_zoom = min(1.0, max(0.45, self.fov_zoom * (1.12 if d > 0 else 0.89)))
        elif self.cam == 'Chase':
            self.orbit[1] = min(7.0, max(1.2, self.orbit[1] * (1.12 if d > 0 else 0.89)))
        else:
            self.over[1] = min(1.5, max(0.4, self.over[1] * (1.1 if d > 0 else 0.91)))

    def why_illegal(self, p):
        s = self.state
        if not s.map.is_open(p):
            return 'That is a boulder field already'
        if s.map.protected(p):
            return 'No-strike zone around the base and pad'
        if manhattan(p, s.rover) <= 1:
            return 'Too close to the rover'
        if p in s.pending:
            return 'Debris is already on its way there'
        if s.blk.get(p) == HARD:
            return 'Already fused'
        return 'Not allowed'

    # ------------------------------------------------------------ per frame
    def tick(self, task):
        now = self.now()
        dt = min(0.05, self.base.taskMgr.globalClock.getDt())
        if self.screen != 'game':
            self.menu_camera(now)
            return task.cont
        s = self.state
        if self.phase == 'play' and not s.result and not self.paused:
            self.step_turns(now)
            self.autopilot()
        if self.dirty:
            self.overlays()
            self.sync_boulders()
            self.build_route()
            self.hud.map_dirty = True
            self.dirty = False
            if self.mode == 'rover':
                self.story_checks()
            if s.result and self.phase == 'play':
                self.begin_outro()
        self.place_rover(dt)
        self.update_meteors(now)
        self.update_ufo(now, dt)
        self.update_effects(now)
        if self.phase == 'intro':
            t = (now - self.phase_t0) / 4.0
            self.intro_camera(t)
            if t >= 1:
                self.end_intro()
        elif self.phase in ('outro', 'debrief') and s.result:
            self.outro_frame(now, final=self.phase == 'debrief')
        else:
            self.update_camera(now)
        self.update_hover()
        pulse = 0.5 + 0.5 * math.sin(now * 3)
        self.pad_glow.setAlphaScale(0.45 + 0.4 * pulse)
        self.base_beacon.setAlphaScale(1.0 if int(now * 2) % 2 else 0.1)
        self.radar.setPos(self.rover_np.getX(), self.rover_np.getY(), self.rover_np.getZ() + 0.06)
        self.shield_np.show() if s.shields > 0 and self.rover_body and not s.result else self.shield_np.hide()
        self.shield_np.setAlphaScale(0.6 + 0.4 * pulse)
        self.space_motion(now, dt)
        for node, kind in self.boulder_nodes.values():     # pulse the alien markers
            if node.hasPythonTag('mark'):
                node.getPythonTag('mark').setAlphaScale(0.35 + 0.65 * pulse)
        moving = self.motion.v > 0.05 or getattr(self.rover, 'thrust', 0) > 0.25
        self.sound.motor(moving)
        if moving and now - self.dust_t > 0.07:
            self.dust_t = now
            self.emit_dust()
        if self.phase != 'intro':
            self.hud.update(now)
        return task.cont

    def space_motion(self, now, dt):
        st0 = getattr(self, 'station', None)
        if st0 is not None and not st0.isEmpty():
            st0.getPythonTag('beacons').setAlphaScale(1.0 if int(now * 1.5) % 2 else 0.15)
        if self.world is not None:
            for node, w in self.world.getPythonTag('spinners') or []:
                node.setHpr(node.getHpr() + w * dt)
            cloud = self.world.getPythonTag('cloud')
            if cloud:
                cloud.setH(now * 0.6)
        st = getattr(self, 'station', None)
        if st is not None and not st.isEmpty():
            st.getPythonTag('ring').setR(now * 6)
        for p, (node, kind) in self.boulder_nodes.items():
            node.setZ(0.06 * math.sin(now * 0.8 + p[0] * 1.7 + p[1]))
            node.setH((p[0] * 53 + p[1] * 31 + now * 8) % 360)

    def story_checks(self):
        s, hud = self.state, self.hud
        if s.result:
            return
        known = self.costs[0]
        if s.rover in s.pending and s.pending[s.rover] == 1 and (s.rover, s.tick) not in self.warned:
            self.warned.add((s.rover, s.tick))
            hud.radio('debris_on_me', once=False)
            self.sound.play('alarm')
        if known == INF:
            hud.radio('no_route')
        elif s.fuel - known < 4 or s.fuel < 0.25 * s.fuel0:
            hud.radio('low_fuel')
        d = manhattan(s.rover, s.map.goal)
        if d <= self.start_dist / 2:
            hud.radio('halfway')
        if d <= 3:
            hud.radio('near_pad')
        if s.budget == 0:
            hud.radio('quiet')

    def step_turns(self, now):
        s = self.state
        if self.motion.remaining() > 0.35 and s.turn == ROVER:
            return
        if self.mode == 'ai':
            if not self.step_once and now < self.next_time:
                return
            self.step_once = False
            if s.turn == ALIEN:
                self.do_alien(self.alien_ai.act(s))
            else:
                self.do_rover(self.rover_ai.act(s.rover_view()))
            self.next_time = self.now() + self.interval()
        elif self.mode == 'alien':
            if s.turn == ROVER and now >= self.next_time:
                self.do_rover(self.rover_ai.act(s.rover_view()))
                self.next_time = self.now() + self.interval()
        elif self.mode == 'rover':
            if s.turn == ALIEN and now >= self.next_time:
                self.do_alien(self.alien_ai.act(s))

    def place_rover(self, dt):
        """Fly the shuttle with inertia: it chases the planned path like a craft
        on thrusters (damped spring), banking from sideways acceleration, dipping
        its nose under thrust and drifting gently when idle."""
        from ui3d.spacelook import Flight
        tx, ty, th = self.motion.update(dt)
        fl = getattr(self, 'flight', None)
        if fl is None or getattr(self, 'flight_state', None) is not self.state:
            fl = self.flight = Flight(tx, ty, th)
            self.flight_state = self.state
        x, y, h, a_fwd, speed = fl.update(tx, ty, th, dt)
        now = self.now()
        self.rover.engines(a_fwd, speed, fl.hacc, now)
        z = 0.05 * math.sin(now * 1.3) + 0.02 * math.sin(now * 3.1)
        idle = max(0.0, 1 - speed * 2)
        self.rover_np.setPosHpr(x, y, z, h + idle * 1.5 * math.sin(now * 0.6),
                                fl.pitch + idle * 1.8 * math.sin(now * 0.9),
                                fl.bank + idle * 2.0 * math.sin(now * 0.7))
        self.rover.bank_angle = fl.bank

    def update_drag(self):
        if self.drag is None:
            return
        m = self.mouse()
        if m is None:
            return
        dx, dy = m[0] - self.drag[0], m[1] - self.drag[1]
        look, orbit, over = self.drag_ref
        if self.cam == 'Pilot':
            self.look = [look[0] - dx * 120, max(-40, min(35, look[1] + dy * 50))]
        elif self.cam == 'Chase':
            self.orbit[0] = orbit[0] - dx * 160
        else:
            self.over[0] = over[0] - dx * 160

    def update_camera(self, now):
        cam = self.base.camera
        lens = self.base.camLens
        self.update_drag()
        rx, ry, rz = self.rover_np.getX(), self.rover_np.getY(), self.rover_np.getZ()
        shake = Vec3(0, 0, 0)
        if self.shake > 0:
            self.shake = max(0, self.shake - 0.02)
            shake = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) * self.shake * 0.06
        if self.cam == 'Pilot':
            if self.drag is None:
                self.look = [self.look[0] * 0.9, self.look[1] * 0.9]
            lens.setFov(80 * self.fov_zoom)
            cam.reparentTo(self.rover_np)
            cam.setPos(Vec3(0, 0.26, 0.33) + shake)       # in the cockpit
            cam.setHpr(self.look[0], -5 + self.look[1], self.rover.bank_angle * 0.6)
            self.rover_body.show()
            self.rover.head.hide()
            return
        self.rover_body.show()
        self.rover.head.show()
        cam.reparentTo(self.base.render)
        n = self.state.map.size
        if self.cam == 'Chase':
            lens.setFov(68)
            h = math.radians(self.rover_np.getH() + self.orbit[0])
            fx, fy = -math.sin(h), math.cos(h)
            dist = self.orbit[1]
            want = Vec3(rx - fx * dist, ry - fy * dist, rz + 0.45 + dist * 0.32)
            look = Vec3(rx + fx * 1.2, ry + fy * 1.2, rz + 0.1)
        else:
            lens.setFov(52)
            h = math.radians(self.over[0])
            back = Vec3(math.sin(h), -math.cos(h), 0) * n * 0.78 * self.over[1]
            centre = Vec3(n / 2, -n / 2, 0)
            want = centre + back + Vec3(0, 0, n * 1.08 * self.over[1])
            look = centre + Vec3(0, 0.6, 0)
        if self.cam_pos is None:
            self.cam_pos, self.cam_look = want, look
        self.cam_pos = self.cam_pos + (want - self.cam_pos) * 0.12
        self.cam_look = self.cam_look + (look - self.cam_look) * 0.15
        cam.setPos(self.cam_pos + shake)
        cam.lookAt(self.cam_look)

    def menu_camera(self, now):
        if not self.menu_nodes:
            return
        if self.world is not None:
            for node, w in self.world.getPythonTag('spinners') or []:
                node.setHpr(node.getHpr() + w * 0.016)
        self.menu_station.getPythonTag('ring').setR(now * 6)
        n = self.menu_map.size
        cam = self.base.camera
        cam.reparentTo(self.base.render)
        a = now * 0.05
        cx, cy = n / 2, -n / 2
        cam.setPos(cx + math.cos(a) * 15, cy + math.sin(a) * 15, 4.5)
        cam.lookAt(cx, cy + 6, 2.5)
        self.base.camLens.setFov(62)
        self.menu_ufo.setPos(cx + math.cos(a * 3) * 6, cy + math.sin(a * 2.2) * 6, 3.2 + 0.2 * math.sin(now * 2))
        self.menu_saucer.animate(now, 2.0)
        if self.screen == 'briefing':
            m = self.brief_mission
            shown = int((now - self.brief_t0) * 70)
            total = 0
            for t, line in zip(self.brief_lines, m.briefing):
                t.setText(line[:max(0, shown - total)])
                total += len(line)

    # ------------------------------------------------------------ world objects
    def sync_boulders(self):
        s = self.state
        rover_view = self.mode == 'rover'
        blk = s.belief if rover_view else s.blk
        for p in list(self.boulder_nodes):
            node, kind = self.boulder_nodes[p]
            if blk.get(p) != kind:
                node.removeNode()
                del self.boulder_nodes[p]
        for p, kind in blk.items():
            if p not in self.boulder_nodes:
                style = self.styles.get(p, 'rock')
                node = models.obstacle_model(style, kind == HARD, (p[0] * 7 + p[1] * 13) % 8, self.glow)
                x, y = self.world_xy(p)
                node.setPos(x, y, self.gz(x, y))
                node.setH((p[0] * 53 + p[1] * 31) % 360)
                node.reparentTo(self.dyn)
                self.boulder_nodes[p] = (node, kind)
            node = self.boulder_nodes[p][0]
            if rover_view and not s.in_radar(p):
                node.setColorScale(0.55, 0.55, 0.65, 1)
            else:
                node.clearColorScale()

    def build_route(self):
        from ui.app import smooth, beads
        self.route_np.getChildren().detach()
        self.cut_np.getChildren().detach()
        self.way_np.getChildren().detach()
        s = self.state
        if self.show_plan and len(self.plan) > 1 and not s.result:
            pts = smooth([self.world_xy(p) for p in self.plan])
            for x, y in beads(pts, 0.3)[2:70]:
                c = self.card(self.route_np, 0.13, (0.5, 0.95, 1, 0.8))
                c.setPos(x, y, self.gz(x, y) + 0.05)
        if self.show_cut and self.cut and not s.result:
            for p in self.cut:
                x, y = self.world_xy(p)
                r = self.meshes.additive(self.meshes.ring('cut', 0.42, 0.48, ORANGE))
                r.setPos(x, y, self.gz(x, y) + 0.04)
                r.reparentTo(self.cut_np)
        if self.waypoint:
            x, y = self.world_xy(self.waypoint)
            ring = self.meshes.additive(self.meshes.ring('way', 0.3, 0.4, GREEN))
            ring.setPos(x, y, self.gz(x, y) + 0.05)
            ring.reparentTo(self.way_np)
            beam = self.card(self.way_np, 0.5, (0.5, 1, 0.6, 0.8))
            beam.setPos(x, y, self.gz(x, y) + 0.6)
            beam.setScale(0.6, 1, 3)

    def alien_choice(self, p, button=1):
        s = self.state
        if button == 3 or self.weapon == 'debris' or s.blk.get(p, OPEN) != OPEN:
            return ('debris', p)
        return ('boulder', p)

    def update_hover(self):
        self.hover_np.getChildren().detach()
        s = self.state
        if self.mode != 'alien' or s.turn != ALIEN or s.result or self.phase != 'play':
            return
        p = self.pick()
        if p is None:
            return
        a = self.alien_choice(p)
        ok = s.legal_alien(a)
        x, y = self.world_xy(p)
        gz = self.gz(x, y)
        r = self.meshes.additive(self.meshes.ring('hover', 0.4, 0.5, GREEN if ok else RED))
        r.setPos(x, y, gz + 0.06)
        r.reparentTo(self.hover_np)
        if ok and a[0] == 'debris':                   # the meteor's path: up, over, and down
            u = Vec3(self.ufo_pos)
            tgt = Vec3(x, y, gz)
            for i in range(1, 24):
                t = i / 24
                pos = u + (tgt - u) * t + Vec3(0, 0, 6 * t * (1 - t))
                c = self.card(self.hover_np, 0.12, (1, 0.5, 0.3, 0.8))
                c.setPos(pos)
        elif ok:
            u = Vec3(self.ufo_pos)
            ls = LineSegs()
            ls.setThickness(2)
            ls.setColor(0.6, 1, 0.8, 0.8)
            ls.moveTo(u)
            ls.drawTo(x, y, gz + 0.3)
            self.meshes.additive(self.hover_np.attachNewNode(ls.create()))

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Autonomous Escape v3 (3D)')
    ap.add_argument('--mode', choices=[m for m, _ in MODES], default='rover')
    ap.add_argument('--rover', choices=list(ROVER_BY_NAME), default='A*')
    ap.add_argument('--alien', choices=list(ALIEN_BY_NAME), default='Core')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--speed', type=int, default=2)
    ap.add_argument('--camera', choices=CAMERAS)
    ap.add_argument('--mission', type=int, help='jump straight into campaign mission 1-5')
    ap.add_argument('--play', action='store_true', help='skip the menu (free play)')
    args = ap.parse_args(argv)
    configure()
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
    game = Game(base, args.mode, args.rover, args.alien, args.seed, args.speed, args.camera)
    if args.mission:
        game.start_mission(args.mission - 1)
    elif args.play:
        game.start_free()
    base.run()
