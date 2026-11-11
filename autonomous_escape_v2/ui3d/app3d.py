"""Autonomous Escape v2: the same game in 3D, seen from the rover's pilot seat.

The rules engine and every AI are shared with the 2D version (package
`escape`); only the presentation differs.

Cameras (V cycles)
  Pilot     sitting on the camera mast, looking where the rover faces
  Chase     behind and above the rover
  Overhead  tactical view of the whole field (best for playing the alien)

Driving (You are the Rover)
  Up / W     drive forward one tile (bump a loose boulder ahead to shoot it)
  Down / S   reverse one tile
  Left/Right turn 90 degrees on the spot (free, takes no turn)
  Space      wait one turn
In Overhead view the arrows drive north / south / west / east instead.
"""
import math
import os
import random
import sys

import numpy as np
from panda3d.core import (CardMaker, Filename, LineSegs, NodePath, Point3, TextNode, TransparencyAttrib, Vec3,
                          WindowProperties, loadPrcFileData)

from escape import GameState, OPEN, SOFT, HARD, ROVER, ALIEN, INF
from escape.agents import ROVER_BY_NAME, ALIEN_BY_NAME, make_rover, make_alien, CoreAlien
from escape.graph import astar, manhattan

W, H = 1280, 780
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui', 'assets')
MODES = [('ai', 'AI vs AI'), ('alien', 'You are the Alien'), ('rover', 'You are the Rover')]
CAMERAS = ['Pilot', 'Chase', 'Overhead']
SPEEDS = [1, 2, 4, 8]
CYAN = (0.34, 0.86, 0.91, 1)
ORANGE = (1, 0.62, 0.25, 1)
RED = (0.94, 0.33, 0.38, 1)
GREEN = (0.43, 0.86, 0.55, 1)
TEXT = (0.93, 0.93, 0.96, 1)
MUTED = (0.62, 0.6, 0.72, 1)
METEOR_DIR = Vec3(0.45, 0.55, 0.72).normalized()   # debris falls in from the north-east sky


def configure(offscreen=False):
    loadPrcFileData('', f'''
        window-title Autonomous Escape v2 - 3D pilot view
        win-size {W} {H}
        framebuffer-multisample 1
        multisamples 4
        sync-video 1
        textures-power-2 none
        show-frame-rate-meter 0
        audio-library-name null
        {'window-type offscreen' if offscreen else ''}
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
    d = (b - a + 180) % 360 - 180
    return a + d * t


class Game:
    def __init__(self, base, mode='rover', rover='A*', alien='Core', seed=None, speed=2, camera=None):
        from ui3d import meshes
        self.base = base
        self.meshes = meshes
        self.mode = mode
        self.rover_name, self.alien_name = rover, alien
        self.seed = seed if seed is not None else random.randrange(1000)
        self.speed = speed
        self.camera_mode = camera
        self.menu_row = 0
        self.show_plan = True
        self.show_cut = False
        self.paused = False
        self.screen_name = 'menu'
        self.glow = meshes.glow_texture()
        self.font = self._font()
        self.game_nodes = []
        base.setBackgroundColor(0.01, 0.01, 0.02, 1)
        base.disableMouse()
        base.camLens.setNearFar(0.03, 2000)
        self.menu_root = None
        self.hud = None
        self.world = None
        self.world_cache = {}
        self.sky = None
        self.keys()
        base.taskMgr.add(self.tick, 'tick')
        self.show_menu()

    # ------------------------------------------------------------ setup
    def _font(self):
        path = 'C:/Windows/Fonts/segoeui.ttf'
        if os.path.exists(path):
            f = self.base.loader.loadFont(Filename.fromOsSpecific(path).getFullpath())
            f.setPixelsPerUnit(60)
            return f
        return None

    def text(self, parent, s, pos, scale=0.05, fg=TEXT, align=TextNode.ALeft, bold=False):
        from direct.gui.OnscreenText import OnscreenText
        t = OnscreenText(text=s, pos=pos, scale=scale, fg=fg, align=align, parent=parent, mayChange=True,
                         font=self.font, shadow=(0, 0, 0, 0.6))
        return t

    def panel(self, parent, l, r, b, t, color=(0.08, 0.07, 0.17, 0.82), edge=(0.23, 0.2, 0.42, 1)):
        from direct.gui.DirectGui import DirectFrame
        f = DirectFrame(parent=parent, frameColor=color, frameSize=(l, r, b, t))
        ls = LineSegs()
        ls.setColor(*edge)
        ls.setThickness(1.5)
        for a, b2 in (((l, b), (r, b)), ((r, b), (r, t)), ((r, t), (l, t)), ((l, t), (l, b))):
            ls.moveTo(a[0], 0, a[1])
            ls.drawTo(b2[0], 0, b2[1])
        f.attachNewNode(ls.create())
        return f

    def keys(self):
        b = self.base
        for k in ('arrow_up', 'arrow_down', 'arrow_left', 'arrow_right', 'w', 'a', 's', 'd'):
            b.accept(k, self.on_key, [k])
            b.accept(k + '-repeat', self.on_key, [k, True])
        for k in ('enter', 'space', 'escape', 'r', 'n', 'v', 'p', 'c', '+', '=', '-', 'm', 'tab'):
            b.accept(k, self.on_key, [k])
        b.accept('mouse1', self.on_click, [1])
        b.accept('mouse3', self.on_click, [3])

    # ------------------------------------------------------------ menu
    def menu_rows(self):
        return [('Mode', dict(MODES)[self.mode], True),
                ('Rover AI', self.rover_name if self.mode != 'rover' else 'You', self.mode != 'rover'),
                ('Alien AI', self.alien_name if self.mode != 'alien' else 'You', self.mode != 'alien'),
                ('Map', f'#{self.seed}', True),
                ('Speed', f'{self.speed} moves / s', self.mode != 'rover'),
                ('Camera', self.default_camera(), True)]

    def default_camera(self):
        if self.camera_mode:
            return self.camera_mode
        return {'rover': 'Pilot', 'ai': 'Chase', 'alien': 'Overhead'}[self.mode]

    def menu_change(self, row, d):
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
            cur = self.default_camera()
            self.camera_mode = CAMERAS[(CAMERAS.index(cur) + d) % len(CAMERAS)]
        self.draw_menu()

    def show_menu(self):
        self.clear_game()
        self.screen_name = 'menu'
        from direct.gui.OnscreenImage import OnscreenImage
        self.menu_root = self.base.aspect2d.attachNewNode('menu')
        bg = OnscreenImage(image=Filename.fromOsSpecific(os.path.join(ASSETS, 'space.jpg')).getFullpath(),
                           parent=self.base.render2d)
        bg.setBin('background', 0)
        self.menu_bg = bg
        self.draw_menu()

    def draw_menu(self):
        from direct.gui.DirectGui import DirectButton
        if self.menu_root is None:
            return
        self.menu_root.getChildren().detach()
        root = self.menu_root
        a = self.base.getAspectRatio()
        self.text(root, 'AUTONOMOUS ESCAPE', (-a + 0.16, 0.74), 0.13, TEXT)
        self.text(root, 'VERSION 2  ·  3D PILOT VIEW', (-a + 0.17, 0.64), 0.05, CYAN)
        self.text(root, 'An asteroid has hit the mining base on Phobos. Drive the last rover across the debris',
                  (-a + 0.17, 0.55), 0.042, MUTED)
        self.text(root, 'field to the launch pad before the alien seals every route.', (-a + 0.17, 0.495), 0.042, MUTED)
        x0, w = -a + 0.16, 1.2
        for i, (label, value, enabled) in enumerate(self.menu_rows()):
            y = 0.33 - i * 0.135
            sel = i == self.menu_row
            self.panel(root, x0, x0 + w, y - 0.055, y + 0.055, edge=CYAN if sel else (0.23, 0.2, 0.42, 1))
            self.text(root, label, (x0 + 0.05, y - 0.018), 0.05, TEXT if enabled else MUTED)
            self.text(root, value, (x0 + 0.86, y - 0.018), 0.05, TEXT if enabled else MUTED, TextNode.ACenter)
            if enabled:
                for dx, ch, d in ((0.62, '<', -1), (1.1, '>', 1)):
                    DirectButton(parent=root, text=ch, text_font=self.font, scale=0.05, pos=(x0 + dx, 0, y - 0.018),
                                 text_fg=CYAN, frameColor=(0.17, 0.16, 0.34, 1), pad=(0.3, 0.1),
                                 command=self.menu_click, extraArgs=[i, d], relief=1)
        y = 0.33 - 6 * 0.135 - 0.02
        DirectButton(parent=root, text='LAUNCH', text_font=self.font, scale=0.065, pos=(x0 + w / 2, 0, y - 0.02),
                     text_fg=(0.04, 0.04, 0.1, 1), frameColor=CYAN, pad=(3.6, 0.35), command=self.start_game,
                     relief=1)
        self.text(root, 'Arrows choose  ·  Enter starts  ·  Esc quits', (x0, y - 0.15), 0.04, MUTED)
        # how it plays
        cx = 0.12
        self.panel(root, cx, a - 0.12, -0.72, 0.39)
        self.text(root, 'In the pilot seat', (cx + 0.05, 0.3), 0.055, TEXT)
        lines = [(CYAN, 'Up / W drive forward, Down / S reverse, Left / Right turn.'),
                 (CYAN, 'Drive into a loose (warm) boulder to blast it: 2 fuel.'),
                 (CYAN, 'Radar shows the alien\'s boulders 5 tiles around.'),
                 (ORANGE, 'Glowing debris streaks across the sky: watch the red'),
                 (ORANGE, 'landing rings. Debris fuses boulders or hits you for 6.'),
                 (ORANGE, 'The saucer strikes every 3 turns, 8 strikes in all.'),
                 (GREEN, 'Reach the pad in the crater. Every move costs 1 fuel.'),
                 (MUTED, 'V camera  ·  Tab map  ·  P route  ·  R replay  ·  Esc menu')]
        for k, (col, line) in enumerate(lines):
            self.text(root, line, (cx + 0.06, 0.19 - k * 0.105), 0.042, col)

    def menu_click(self, row, d):
        self.menu_row = row
        self.menu_change(row, d)

    # ------------------------------------------------------------ game
    def clear_game(self):
        if self.menu_root is not None:
            self.menu_root.removeNode()
            self.menu_root = None
            self.menu_bg.destroy()
        for n in self.game_nodes:
            n.removeNode()
        self.game_nodes = []
        if self.world is not None:
            self.world.detachNode()
            self.world = None
        self.base.render.clearLight()

    def start_game(self, new_seed=False):
        from ui3d import world as W3
        if new_seed:
            self.seed = random.randrange(1000)
        self.clear_game()
        self.screen_name = 'game'
        self.cam = self.default_camera()
        s = self.state = GameState.new(self.seed)
        self.rover_ai = make_rover(self.rover_name, self.seed) if self.mode != 'rover' else None
        self.alien_ai = make_alien(self.alien_name, self.seed, **self._alien_kw()) if self.mode != 'alien' else None
        for ag in (self.rover_ai, self.alien_ai):
            if ag:
                ag.reset()
        self.core_view = CoreAlien()
        render = self.base.render
        if self.seed not in self.world_cache:
            if len(self.world_cache) > 2:
                for w in self.world_cache.values():
                    w[0].removeNode()
                self.world_cache.clear()
            self.world_cache[self.seed] = W3.build_world(render, s.map, self.glow)
        self.world, self.ground, self.fields = self.world_cache[self.seed]
        self.world.reparentTo(render)
        if self.sky is None:
            self.sky = W3.build_sky(self.base, self.glow)
        self.sun = W3.lights(self.base, render, s.map.size)
        self.setup_shadows()

        def keep(np_):
            self.game_nodes.append(np_)
            return np_
        self.keep = keep
        self.dyn = keep(render.attachNewNode('dynamic'))
        self.rover_np = self.dyn.attachNewNode('rover_pose')     # the camera can ride on this
        self.rover_body = W3.rover_model()
        self.rover_body.reparentTo(self.rover_np)
        self.ufo = W3.ufo_model(self.glow)
        self.ufo.reparentTo(self.dyn)
        self.ufo_pos = Vec3(s.map.size / 2, -s.map.size / 2, 3.5)
        self.ufo_goal = None
        self.ufo_until = 0
        self.pad = self.make_pad(s.map.goal)
        self.base_station = self.make_base_station(s.map.start)
        self.radar = self.meshes.additive(self.meshes.ring('radar', s.rules.radar + 0.42, s.rules.radar + 0.5, CYAN))
        self.radar.setAlphaScale(0.5)
        self.radar.reparentTo(self.dyn)
        self.route_np = self.dyn.attachNewNode('route')
        self.cut_np = self.dyn.attachNewNode('cut')
        self.hover_np = self.dyn.attachNewNode('hover')
        self.boulder_nodes = {}
        self.meteors = {}
        self.effects = []
        self.heading = self.initial_heading()
        h = heading_of(self.heading)
        self.anim = (s.rover, s.rover, 0.0, 0.01, h, h)
        self.last_tick = self.now()
        self.next_time = self.now() + 1.0
        self.cam_pos = None
        self.shake = 0
        self.dirty = True
        self.overlays()
        self.build_hud()
        self.place_rover(1.0)

    def _alien_kw(self):
        return {'time_limit': 0.5} if self.alien_name == 'MCTS' else {}

    def initial_heading(self):
        s = self.state
        _, plan = astar(s.map, {}, s.rover, s.map.goal, s.rules)
        if len(plan) > 1:
            return (plan[1][0] - plan[0][0], plan[1][1] - plan[0][1])
        return (1, 0)

    def setup_shadows(self):
        try:
            from panda3d.core import CullFaceAttrib, RenderState
            node = self.sun.node()
            node.setShadowCaster(True, 4096, 4096)
            # draw back faces into the shadow map: no self-shadow speckles on rocks
            node.setInitialState(RenderState.make(CullFaceAttrib.makeReverse()))
            self.base.render.setShaderAuto()
        except Exception:      # older drivers: plain lighting still works
            pass

    def now(self):
        return self.base.taskMgr.globalClock.getFrameTime()

    def world_xy(self, p):
        return float(p[1]), float(-p[0])

    def gz(self, x, y):
        return self.ground.z(x, y)

    def make_pad(self, p):
        x, y = self.world_xy(p)
        root = self.dyn.attachNewNode('pad')
        rim = max(self.gz(x + 0.6 * math.cos(a), y + 0.6 * math.sin(a)) for a in np.linspace(0, 6.28, 16))
        root.setPos(x, y, max(rim, self.gz(x, y)) + 0.04)
        base = self.meshes.disc('padbase', 0.62, (0.12, 0.15, 0.2, 1))
        base.reparentTo(root)
        for r_in, r_out, col in ((0.56, 0.62, CYAN), (0.4, 0.45, (0.9, 0.97, 1, 1))):
            ring = self.meshes.ring('padring', r_in, r_out, col)
            ring.setZ(0.005)
            self.meshes.additive(ring)
            ring.reparentTo(root)
        for dx in (-0.18, 0.18):
            self.meshes.box('H', 0.06, 0.4, 0.01, (0.9, 0.97, 1, 1), (dx, 0, 0.01)).reparentTo(root)
        self.meshes.box('H', 0.36, 0.06, 0.01, (0.9, 0.97, 1, 1), (0, 0, 0.01)).reparentTo(root)
        self.pad_glow = self.card(root, 2.6, CYAN[:3] + (0.5,))
        self.pad_glow.setZ(0.1)
        self.pad_beacon = self.card(root, 0.5, (0.6, 1, 1, 1))
        self.pad_beacon.setZ(1.2)
        mast = self.meshes.box('padmast', 0.04, 0.04, 1.2, (0.8, 0.85, 0.9, 1), (0.66, 0.66, 0.6))
        mast.reparentTo(root)
        return root

    def make_base_station(self, p):
        x, y = self.world_xy(p)
        root = self.dyn.attachNewNode('base')
        root.setPos(x - 2.2, y + 2.2, 0.02)
        root.setShaderOff(1)
        dome = self.meshes.sphere('dome', (0.88, 0.89, 0.92, 1), sub=3)
        dome.setScale(0.55, 0.55, 0.42)
        dome.reparentTo(root)
        self.meshes.box('panel', 1.1, 0.35, 0.03, (0.15, 0.28, 0.65, 1), (0.9, 0.3, 0.5)).reparentTo(root)
        self.meshes.box('leg', 0.05, 0.05, 0.5, (0.7, 0.7, 0.75, 1), (0.9, 0.3, 0.25)).reparentTo(root)
        self.base_beacon = self.card(root, 0.35, (1, 0.25, 0.25, 1))
        self.base_beacon.setZ(0.5)
        return root

    def card(self, parent, size, color):
        cm = CardMaker('glow')
        cm.setFrame(-size / 2, size / 2, -size / 2, size / 2)
        np_ = parent.attachNewNode(cm.generate())
        np_.setTexture(self.glow)
        np_.setColor(*color)
        np_.setBillboardPointEye()
        self.meshes.additive(np_)
        return np_

    # ------------------------------------------------------------ HUD
    def build_hud(self):
        from direct.gui.OnscreenImage import OnscreenImage
        b = self.base
        self.hud = self.keep(b.aspect2d.attachNewNode('hud'))
        tl = self.keep(b.a2dTopLeft.attachNewNode('tl'))
        tr = self.keep(b.a2dTopRight.attachNewNode('tr'))
        bl = self.keep(b.a2dBottomLeft.attachNewNode('bl'))
        br = self.keep(b.a2dBottomRight.attachNewNode('br'))
        self.panel(tl, 0.03, 0.95, -0.47, -0.03)
        rv = self.rover_ai.name if self.rover_ai else 'You'
        al = self.alien_ai.name if self.alien_ai else 'You'
        self.t_title = self.text(tl, f'ROVER {rv}   vs   ALIEN {al}', (0.07, -0.1), 0.045, TEXT)
        self.text(tl, 'FUEL', (0.07, -0.18), 0.038, CYAN)
        self.fuel_bg = self.panel(tl, 0.07, 0.91, -0.235, -0.205, color=(0.16, 0.15, 0.28, 1))
        from direct.gui.DirectGui import DirectFrame
        self.fuel_bar = DirectFrame(parent=tl, frameColor=GREEN, frameSize=(0, 0.84, -0.235, -0.205), pos=(0.07, 0, 0))
        self.t_fuel = self.text(tl, '', (0.91, -0.18), 0.04, TEXT, TextNode.ARight)
        self.t_route = self.text(tl, '', (0.07, -0.29), 0.038, TEXT)
        self.text(tl, 'ALIEN', (0.07, -0.36), 0.038, ORANGE)
        self.pips = []
        for i in range(self.state.rules.alien_budget):
            p = DirectFrame(parent=tl, frameColor=ORANGE, frameSize=(-0.016, 0.016, -0.016, 0.016),
                            pos=(0.22 + i * 0.05, 0, -0.348))
            self.pips.append(p)
        self.t_alien = self.text(tl, '', (0.07, -0.43), 0.038, TEXT)
        # AI + log
        self.panel(bl, 0.03, 1.05, 0.03, 0.42)
        self.t_ai = self.text(bl, '', (0.07, 0.36), 0.036, (0.75, 0.62, 1, 1))
        self.t_log = [self.text(bl, '', (0.07, 0.29 - i * 0.05), 0.034, MUTED) for i in range(5)]
        # controls
        self.t_help = self.text(br, '', (-0.05, 0.1), 0.034, MUTED, TextNode.ARight)
        # minimap
        self.map_size = 240
        self.map_tex = None
        self.map_parent = tr
        self.map_img = None
        self.map_frame = self.panel(tr, -0.61, -0.03, -0.61, -0.03, color=(0, 0, 0, 0), edge=CYAN)
        self.t_cam = self.text(tr, '', (-0.32, -0.67), 0.036, CYAN, TextNode.ACenter)
        # centre: pilot reticle, banner, end screen
        self.reticle = self.hud.attachNewNode('reticle')
        ls = LineSegs()
        ls.setColor(*CYAN[:3], 0.7)
        ls.setThickness(1.5)
        for sx, sy in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
            ls.moveTo(sx * 0.42, 0, sy * 0.3)
            ls.drawTo(sx * 0.42 - sx * 0.07, 0, sy * 0.3)
            ls.moveTo(sx * 0.42, 0, sy * 0.3)
            ls.drawTo(sx * 0.42, 0, sy * 0.3 - sy * 0.07)
        ls.moveTo(-0.025, 0, 0)
        ls.drawTo(0.025, 0, 0)
        ls.moveTo(0, 0, -0.025)
        ls.drawTo(0, 0, 0.025)
        self.reticle.attachNewNode(ls.create())
        self.t_heading = self.text(self.reticle, '', (0, 0.36), 0.045, CYAN, TextNode.ACenter)
        self.t_banner = self.text(self.hud, '', (0, 0.62), 0.05, (1, 0.85, 0.75, 1), TextNode.ACenter)
        self.t_flash = self.text(self.hud, '', (0, -0.55), 0.048, (1, 0.7, 0.7, 1), TextNode.ACenter)
        self.flash_t = -10
        self.red = DirectFrame(parent=b.render2d, frameColor=(1, 0, 0, 0), frameSize=(-1, 1, -1, 1))
        self.keep(self.red)
        self.end_root = None
        self.update_minimap()

    def say(self, msg):
        self.t_flash.setText(msg)
        self.flash_t = self.now()

    def update_hud(self):
        s = self.state
        frac = max(0, s.fuel) / s.fuel0
        self.fuel_bar['frameSize'] = (0, 0.84 * frac, -0.235, -0.205)
        self.fuel_bar['frameColor'] = GREEN if frac > 0.45 else (0.98, 0.82, 0.35, 1) if frac > 0.2 else RED
        self.t_fuel.setText(f'{max(s.fuel, 0)} / {s.fuel0}')
        rover_view = self.mode == 'rover'
        cost = self.costs[0] if rover_view else self.costs[1]
        label = 'Best known route' if rover_view else 'Cheapest true route'
        self.t_route.setText(f'{label}: ' + ('none' if cost == INF else f'{cost:.0f} fuel'))
        self.t_route['fg'] = TEXT if cost != INF and cost <= s.fuel else RED
        for i, p in enumerate(self.pips):
            p['frameColor'] = ORANGE if i < s.budget else (0.3, 0.26, 0.3, 1)
        if s.result:
            msg = ''
        elif s.turn == ALIEN:
            msg = 'YOUR STRIKE: click the ground (Space holds)' if self.mode == 'alien' else 'The saucer is striking...'
        elif s.budget == 0:
            msg = 'Alien is out of strikes'
        else:
            k = s.rules.alien_every - s.tick % s.rules.alien_every
            msg = f'Turn {s.tick}  ·  next strike in {k}'
        if s.pending and not s.result:
            msg += f'  ·  {len(s.pending)} debris incoming'
        self.t_alien.setText(msg)
        lines = []
        if self.rover_ai:
            lines.append(f'{self.rover_ai.name}: ' + (self.rover_ai.info or f'route {max(0, len(self.plan) - 1)} tiles'))
        if self.alien_ai and not rover_view:
            lines.append(f'{self.alien_ai.name}: ' + (self.alien_ai.info or 'waiting'))
        self.t_ai.setText('   ·   '.join(lines)[:90])
        events = s.log if not rover_view else [(t, m) for t, m in s.log if 'drops a boulder' not in m]
        ev = list(reversed(events[-5:]))
        for i, t in enumerate(self.t_log):
            t.setText(f'T{ev[i][0]:>3}  {ev[i][1]}' if i < len(ev) else '')
        if self.mode == 'rover':
            help_ = ('Up/W forward · Down/S reverse · Left/Right turn · Space wait' if self.cam != 'Overhead'
                     else 'Arrows drive N/S/W/E · Space wait')
        elif self.mode == 'alien':
            help_ = 'Left-click boulder (or debris on a boulder) · Right-click debris · Space hold'
        else:
            help_ = f'Space pause{" (paused)" if self.paused else ""} · → step · +/- speed ({self.speed}/s)'
        self.t_help.setText(help_ + '\nV camera · Tab map · P route · C cut · R replay · N new · Esc menu')
        self.t_cam.setText(f'{self.cam.upper()} CAM')
        self.reticle.show() if self.cam == 'Pilot' else self.reticle.hide()
        names = {(-1, 0): 'N', (1, 0): 'S', (0, 1): 'E', (0, -1): 'W'}
        self.t_heading.setText(f'HEADING {names.get(self.heading, "")}')
        route = set(self.plan[1:8])
        soon = [p for p in s.pending if p in route or p == s.rover]
        if s.result or not soon:
            self.t_banner.setText('')
        elif s.rover in s.pending:
            self.t_banner.setText('DEBRIS LANDING ON THE ROVER: MOVE!')
        else:
            self.t_banner.setText('Debris incoming on the planned route')
        if self.now() - self.flash_t > 1.8:
            self.t_flash.setText('')

    def update_minimap(self):
        """Top-down map: the 2D terrain image plus markers, drawn with pygame."""
        import pygame as pg
        s = self.state
        n = s.map.size
        if getattr(self, '_map_base_seed', None) != self.seed:
            img = self.fields['image']
            surf = pg.surfarray.make_surface(img.transpose(1, 0, 2))
            self._map_base = pg.transform.smoothscale(surf, (self.map_size, self.map_size))
            self._map_base_seed = self.seed
        m = self._map_base.copy()
        k = self.map_size / n

        def c(p):
            return ((p[1] + 0.5) * k, (p[0] + 0.5) * k)
        rover_view = self.mode == 'rover'
        if len(self.plan) > 1:
            pg.draw.lines(m, (90, 225, 235), False, [c(p) for p in self.plan], 2)
        blk = s.belief if rover_view else s.blk
        for p, b in blk.items():
            pg.draw.circle(m, (230, 150, 90) if b == SOFT else (255, 90, 40), c(p), k * 0.38)
        for p in s.pending:
            pg.draw.circle(m, (255, 90, 80), c(p), k * 0.5, 2)
        pg.draw.circle(m, (90, 225, 235), c(s.map.goal), k * 0.7, 2)
        R = (s.rules.radar + 0.5) * k
        pg.draw.circle(m, (90, 225, 235), c(s.rover), R, 1)
        x, y = c(s.rover)
        dr, dc = self.heading
        tip = (x + dc * k * 0.9, y + dr * k * 0.9)
        side = (-dr * k * 0.5, dc * k * 0.5)
        pg.draw.polygon(m, (255, 255, 255), [tip, (x - dc * k * 0.4 + side[0], y - dr * k * 0.4 + side[1]),
                                             (x - dc * k * 0.4 - side[0], y - dr * k * 0.4 - side[1])])
        arr = pg.surfarray.array3d(m).transpose(1, 0, 2)
        alpha = np.full(arr.shape[:2] + (1,), 235, np.uint8)
        tex = self.meshes.texture(np.concatenate([arr, alpha], -1), 'minimap', mipmap=False)
        if self.map_img is None:
            from direct.gui.OnscreenImage import OnscreenImage
            self.map_img = OnscreenImage(image=tex, parent=self.map_parent, pos=(-0.32, 0, -0.32), scale=0.28)
        else:
            self.map_img.setImage(tex)
        self.map_img.setTransparency(TransparencyAttrib.MAlpha)

    # ------------------------------------------------------------ turns
    def interval(self):
        return 1.0 / self.speed

    def overlays(self):
        s = self.state
        v = s.rover_view()
        _, self.plan = astar(s.map, v.belief, s.rover, s.map.goal, s.rules)
        if self.mode != 'rover':
            self.cut = self.core_view.plan(s)[0]
        else:
            self.cut = []
        known, _ = astar(s.map, v.belief, s.rover, s.map.goal, s.rules)
        self.costs = (known, s.route_cost())

    def anim_dur(self):
        return 0.32 if self.mode == 'rover' else max(0.12, min(0.4, self.interval() * 0.8))

    def do_rover(self, action):
        s = self.state
        landing = [p for p, t in s.pending.items() if t == 1]
        hits = s.hits
        frm = s.rover
        h0 = self.current_heading()
        s.apply_rover(action)
        now = self.now()
        self.last_tick = now
        if action[0] == 'move' and s.rover != frm:
            d = (s.rover[0] - frm[0], s.rover[1] - frm[1])
            if self.mode != 'rover' or self.cam == 'Overhead':
                self.heading = d
            self.anim = (frm, s.rover, now, self.anim_dur(), h0, heading_of(self.heading))
            self.add_dust(frm)
        elif action[0] == 'shoot':
            d = (action[1][0] - s.rover[0], action[1][1] - s.rover[1])
            self.heading = d
            self.anim = (s.rover, s.rover, now, 0.15, h0, heading_of(d))
            self.add_laser(action[1])
        for p in landing:
            self.add_impact(p)
        if s.hits > hits:
            self.shake = 0.5
            self.red['frameColor'] = (1, 0, 0, 0.35)
            self.say(f'Debris hit! -{s.rules.debris_damage} fuel')
        self.dirty = True
        if s.turn == ALIEN and self.mode == 'rover':
            self.next_time = now + 0.5

    def do_alien(self, action):
        s = self.state
        s.apply_alien(action)
        now = self.now()
        if action:
            kind, p = action
            visible = self.mode != 'rover' or s.in_radar(p)
            if visible:
                x, y = self.world_xy(p)
                self.ufo_goal = Vec3(x, y, self.gz(x, y) + 2.4)
                self.ufo_until = now + 1.2
            if kind == 'boulder' and visible:
                self.add_drop(p)
            elif kind == 'debris':
                self.add_launch()
        self.dirty = True
        self.next_time = now + self.interval()

    def current_heading(self):
        return self.rover_np.getH() if hasattr(self, 'rover_np') else 0

    def busy(self):
        frm, to, t0, dur, _, _ = self.anim
        return self.now() - t0 < dur

    def drive(self, key):
        s = self.state
        if s.result or s.turn != ROVER or self.busy():
            return
        absolute = {'arrow_up': (-1, 0), 'w': (-1, 0), 'arrow_down': (1, 0), 's': (1, 0),
                    'arrow_left': (0, -1), 'a': (0, -1), 'arrow_right': (0, 1), 'd': (0, 1)}
        if self.cam == 'Overhead':
            self.heading = absolute[key]
            self.try_move(self.heading)
            return
        dr, dc = self.heading
        if key in ('arrow_left', 'a', 'arrow_right', 'd'):
            self.heading = (-dc, dr) if key in ('arrow_left', 'a') else (dc, -dr)
            h0 = self.current_heading()
            self.anim = (s.rover, s.rover, self.now(), 0.22, h0, heading_of(self.heading))
            self.dirty = True
            return
        d = (dr, dc) if key in ('arrow_up', 'w') else (-dr, -dc)
        self.try_move(d)

    def try_move(self, d):
        s = self.state
        tgt = (s.rover[0] + d[0], s.rover[1] + d[1])
        if not s.map.is_open(tgt):
            self.say('Boulder field: no way through')
            return
        b = s.blk.get(tgt, OPEN)
        if b == HARD:
            self.say('Fused boulder: the laser cannot break it')
            return
        self.do_rover(('shoot', tgt) if b == SOFT else ('move', tgt))

    # ------------------------------------------------------------ input
    def on_key(self, k, repeat=False):
        if self.screen_name == 'menu':
            if repeat:
                return
            if k in ('arrow_up', 'w'):
                self.menu_row = (self.menu_row - 1) % 6
                self.draw_menu()
            elif k in ('arrow_down', 's'):
                self.menu_row = (self.menu_row + 1) % 6
                self.draw_menu()
            elif k in ('arrow_left', 'a'):
                self.menu_change(self.menu_row, -1)
            elif k in ('arrow_right', 'd'):
                self.menu_change(self.menu_row, 1)
            elif k in ('enter', 'space'):
                self.start_game()
            elif k == 'escape':
                sys.exit(0)
            return
        s = self.state
        if k == 'escape':
            self.show_menu()
        elif k == 'r':
            self.start_game()
        elif k == 'n':
            self.start_game(new_seed=True)
        elif k == 'v':
            self.cam = CAMERAS[(CAMERAS.index(self.cam) + 1) % len(CAMERAS)]
            self.cam_pos = None
        elif k == 'p':
            self.show_plan = not self.show_plan
            self.dirty = True
        elif k == 'c':
            self.show_cut = not self.show_cut
            self.dirty = True
        elif k == 'tab' or k == 'm':
            if self.map_img.isHidden():
                self.map_img.show()
                self.map_frame.show()
            else:
                self.map_img.hide()
                self.map_frame.hide()
        elif k in ('+', '='):
            self.speed = SPEEDS[min(len(SPEEDS) - 1, SPEEDS.index(self.speed) + 1)] if self.speed in SPEEDS else 2
        elif k == '-':
            self.speed = SPEEDS[max(0, SPEEDS.index(self.speed) - 1)] if self.speed in SPEEDS else 2
        elif self.mode == 'rover':
            if k == 'space' and s.turn == ROVER and not s.result and not self.busy():
                self.do_rover(('wait', None))
            elif k in ('arrow_up', 'w', 'arrow_down', 's', 'arrow_left', 'a', 'arrow_right', 'd'):
                self.drive(k)
        elif k == 'space':
            if self.mode == 'alien' and s.turn == ALIEN and not s.result:
                self.do_alien(None)
            else:
                self.paused = not self.paused
        elif k == 'arrow_right' and self.mode == 'ai' and not repeat:
            self.paused = True
            self.step_once = True

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
        if abs(a.z - b.z) < 1e-6:
            return None
        t = (a.z - 0.0) / (a.z - b.z)
        p = a + (b - a) * t
        r, c = int(round(-p.y)), int(round(p.x))
        n = self.state.map.size
        return (r, c) if 0 <= r < n and 0 <= c < n else None

    def on_click(self, button):
        if self.screen_name != 'game' or self.mode != 'alien':
            return
        s = self.state
        p = self.pick()
        if p is None or s.result or s.turn != ALIEN:
            return
        a = ('debris', p) if button == 3 else (('boulder', p) if s.blk.get(p, OPEN) == OPEN else ('debris', p))
        if s.legal_alien(a):
            self.do_alien(a)
        else:
            self.say(self.why_illegal(p))

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
        if self.screen_name != 'game':
            return task.cont
        s = self.state
        now = self.now()
        if not s.result:
            self.step_turns(now)
        if self.dirty:
            self.overlays()
            self.sync_boulders()
            self.build_route()
            self.update_minimap()
            self.dirty = False
            if s.result and self.end_root is None:
                self.show_end()
        self.place_rover(now)
        self.update_meteors(now)
        self.update_ufo(now)
        self.update_effects(now)
        self.update_camera(now)
        self.update_hover()
        pulse = 0.5 + 0.5 * math.sin(now * 3)
        self.pad_glow.setAlphaScale(0.5 + 0.4 * pulse)
        self.pad_beacon.setScale(0.8 + 0.4 * pulse)
        self.base_beacon.setAlphaScale(1.0 if int(now * 2) % 2 else 0.1)
        self.radar.setPos(self.rover_np.getX(), self.rover_np.getY(), self.rover_np.getZ() + 0.06)
        a = self.red['frameColor'][3]
        if a > 0:
            self.red['frameColor'] = (1, 0, 0, max(0, a - 0.02))
        self.update_hud()
        return task.cont

    def step_turns(self, now):
        s = self.state
        if self.mode == 'ai':
            step = getattr(self, 'step_once', False)
            if (self.paused and not step) or (not step and now < self.next_time) or self.busy():
                return
            self.step_once = False
            if s.turn == ALIEN:
                self.do_alien(self.alien_ai.act(s))
            else:
                self.do_rover(self.rover_ai.act(s.rover_view()))
            self.next_time = self.now() + self.interval()
        elif self.mode == 'alien':
            if s.turn == ROVER and not self.paused and now >= self.next_time and not self.busy():
                self.do_rover(self.rover_ai.act(s.rover_view()))
                self.next_time = self.now() + self.interval()
        elif self.mode == 'rover':
            if s.turn == ALIEN and now >= self.next_time:
                self.do_alien(self.alien_ai.act(s))

    def place_rover(self, now):
        frm, to, t0, dur, h0, h1 = self.anim
        t = ease((now - t0) / dur)
        (x0, y0), (x1, y1) = self.world_xy(frm), self.world_xy(to)
        x, y = lerp(x0, x1, t), lerp(y0, y1, t)
        h = angle_lerp(h0, h1, t)
        fx, fy = -math.sin(math.radians(h)), math.cos(math.radians(h))
        zf, zb = self.gz(x + fx * 0.25, y + fy * 0.25), self.gz(x - fx * 0.25, y - fy * 0.25)
        zl, zr = self.gz(x - fy * 0.22, y + fx * 0.22), self.gz(x + fy * 0.22, y - fx * 0.22)
        z = max(self.gz(x, y), (zf + zb) / 2)
        pitch = math.degrees(math.atan2(zf - zb, 0.5))
        roll = math.degrees(math.atan2(zl - zr, 0.44))
        bob = 0.012 * math.sin(now * 40) if 0 < t < 1 else 0
        self.rover_np.setPosHpr(x, y, z + bob, h, pitch, roll)

    def update_camera(self, now):
        cam = self.base.camera
        lens = self.base.camLens
        rx, ry, rz = self.rover_np.getX(), self.rover_np.getY(), self.rover_np.getZ()
        h = self.rover_np.getH()
        fx, fy = -math.sin(math.radians(h)), math.cos(math.radians(h))
        shake = Vec3(0, 0, 0)
        if self.shake > 0:
            self.shake = max(0, self.shake - 0.02)
            shake = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) * self.shake * 0.06
        if self.cam == 'Pilot':
            lens.setFov(80)
            cam.reparentTo(self.rover_np)
            cam.setPos(Vec3(0, 0.16, 0.62) + shake)
            cam.setHpr(0, -9, 0)
            self.rover_body.hide()
            return
        self.rover_body.show()
        cam.reparentTo(self.base.render)
        n = self.state.map.size
        if self.cam == 'Chase':
            lens.setFov(68)
            want = Vec3(rx - fx * 2.1, ry - fy * 2.1, rz + 1.45)
            look = Vec3(rx + fx * 1.4, ry + fy * 1.4, rz + 0.1)
        else:
            lens.setFov(52)
            want = Vec3(n / 2, -n / 2 - n * 0.78, n * 1.08)
            look = Vec3(n / 2, -n / 2 + 0.6, 0)
        if self.cam_pos is None:
            self.cam_pos = want
            self.cam_look = look
        self.cam_pos = self.cam_pos + (want - self.cam_pos) * 0.12
        self.cam_look = self.cam_look + (look - self.cam_look) * 0.15
        cam.setPos(self.cam_pos + shake)
        cam.lookAt(self.cam_look)

    # ------------------------------------------------------------ world objects
    def sync_boulders(self):
        from ui3d import world as W3
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
                node = W3.boulder_model(kind, (p[0] * 7 + p[1] * 13) % 8, self.glow)
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
        s = self.state
        if self.show_plan and len(self.plan) > 1 and not s.result:
            pts = smooth([self.world_xy(p) for p in self.plan])
            for x, y in beads(pts, 0.3)[2:70]:
                c = self.card(self.route_np, 0.13, (0.5, 0.95, 1, 0.8))
                c.setPos(x, y, self.gz(x, y) + 0.05)
        if self.show_cut and self.cut and not s.result:
            for p in self.cut:
                x, y = self.world_xy(p)
                r = self.meshes.ring('cut', 0.42, 0.48, ORANGE)
                self.meshes.additive(r)
                r.setPos(x, y, self.gz(x, y) + 0.04)
                r.reparentTo(self.cut_np)

    def update_meteors(self, now):
        s = self.state
        for p in list(self.meteors):
            if p not in s.pending:
                for n in self.meteors[p]:
                    n.removeNode()
                del self.meteors[p]
        prog = min(1.0, (now - self.last_tick) / 0.7) * 0.7
        for p, ttl in s.pending.items():
            x, y = self.world_xy(p)
            gz = self.gz(x, y)
            if p not in self.meteors:
                head = self.dyn.attachNewNode('meteor')
                self.card(head, 0.5, (1, 0.95, 0.8, 1))
                self.card(head, 1.6, (1, 0.55, 0.15, 0.8))
                for i in range(1, 10):
                    c = self.card(head, 0.9 - i * 0.07, (1, 0.5, 0.15, 0.55 - i * 0.05))
                    c.setPos(METEOR_DIR * i * 0.22)
                ring = self.meshes.additive(self.meshes.ring('target', 0.34, 0.42, RED))
                ring.reparentTo(self.dyn)
                ring.setPos(x, y, gz + 0.05)
                dot = self.meshes.additive(self.meshes.ring('target', 0.08, 0.14, RED))
                dot.reparentTo(ring)
                self.meteors[p] = (head, ring)
            head, ring = self.meteors[p]
            d = 1.2 + 3.4 * (ttl - prog)
            head.setPos(Vec3(x, y, gz) + METEOR_DIR * d)
            ring.setScale(1 + 0.15 * math.sin(now * 8))

    def update_ufo(self, now):
        n = self.state.map.size
        if self.ufo_goal is not None and now < self.ufo_until:
            goal = self.ufo_goal
        else:
            goal = Vec3(n * (0.5 + 0.36 * math.sin(now / 5.2)), -n * (0.5 + 0.32 * math.sin(now / 3.9 + 1.3)), 3.2)
        self.ufo_pos += (goal - self.ufo_pos) * 0.04
        self.ufo.setPos(self.ufo_pos + Vec3(0, 0, 0.08 * math.sin(now * 2.4)))
        self.ufo.setH(now * 40)

    # ------------------------------------------------------------ effects
    def add_effect(self, node, dur, fn):
        self.effects.append((node, self.now(), dur, fn))

    def update_effects(self, now):
        keep = []
        for node, t0, dur, fn in self.effects:
            t = (now - t0) / dur
            if t >= 1:
                node.removeNode()
                continue
            fn(node, t)
            keep.append((node, t0, dur, fn))
        self.effects = keep
        hidden = {e[0].getPythonTag('tile') for e in self.effects if e[0].hasPythonTag('tile')}
        for p, (node, _) in self.boulder_nodes.items():
            node.hide() if p in hidden else node.show()

    def add_laser(self, p):
        s = self.state
        x0, y0 = self.world_xy(s.rover)
        x1, y1 = self.world_xy(p)
        a = Vec3(x0, y0, self.gz(x0, y0) + 0.5)
        b = Vec3(x1, y1, self.gz(x1, y1) + 0.22)
        root = self.dyn.attachNewNode('laser')
        ls = LineSegs()
        ls.setThickness(5)
        ls.setColor(1, 0.3, 0.45, 1)
        ls.moveTo(a)
        ls.drawTo(b)
        beam = root.attachNewNode(ls.create())
        self.meshes.additive(beam)
        flash = self.card(root, 1.2, (1, 0.6, 0.4, 1))
        flash.setPos(b)

        def fn(node, t):
            node.setAlphaScale(1 - t)
            flash.setScale(1 + 2 * t)
        self.add_effect(root, 0.4, fn)
        self.add_burst(b, (0.9, 0.55, 0.4, 1), 0.6)

    def add_burst(self, pos, color, dur):
        root = self.dyn.attachNewNode('burst')
        bits = []
        for k in range(10):
            c = self.card(root, 0.12, color)
            v = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.6, 1.6))
            bits.append((c, v))

        def fn(node, t):
            for c, v in bits:
                c.setPos(pos + v * t * 0.9 - Vec3(0, 0, 1.2 * t * t))
            node.setAlphaScale(1 - t)
        self.add_effect(root, dur, fn)

    def add_impact(self, p):
        x, y = self.world_xy(p)
        pos = Vec3(x, y, self.gz(x, y) + 0.15)
        root = self.dyn.attachNewNode('impact')
        flash = self.card(root, 2.2, (1, 0.7, 0.35, 1))
        flash.setPos(pos)
        ring = self.meshes.ring('dust', 0.7, 1.0, (0.75, 0.68, 0.6, 0.6))
        self.meshes.translucent(ring)
        ring.reparentTo(root)
        ring.setPos(pos)

        def fn(node, t):
            flash.setScale(max(0.01, 1 - t * 1.6))
            ring.setScale(0.3 + 1.6 * t)
            ring.setAlphaScale(1 - t)
        self.add_effect(root, 0.8, fn)
        self.add_burst(pos, (0.8, 0.7, 0.6, 1), 0.9)

    def add_dust(self, p):
        x, y = self.world_xy(p)
        root = self.dyn.attachNewNode('dust')
        puffs = []
        for k in range(4):
            c = self.card(root, 0.35, (0.55, 0.48, 0.42, 0.35))
            c.setPos(x + random.uniform(-0.2, 0.2), y + random.uniform(-0.2, 0.2), self.gz(x, y) + 0.1)
            puffs.append(c)

        def fn(node, t):
            for c in puffs:
                c.setScale(1 + 2 * t)
                c.setZ(c.getZ() + 0.004)
            node.setAlphaScale(1 - t)
        self.add_effect(root, 0.9, fn)

    def add_drop(self, p):
        from ui3d import world as W3
        x, y = self.world_xy(p)
        gz = self.gz(x, y)
        root = self.dyn.attachNewNode('drop')
        root.setPythonTag('tile', p)
        rock = W3.boulder_model(SOFT, (p[0] * 7 + p[1] * 13) % 8, self.glow)
        rock.reparentTo(root)
        beam = self.meshes.cone('beam', 0.25, 0.6, 2.3, (1, 0.9, 0.5, 0.18))
        self.meshes.additive(beam)
        beam.reparentTo(root)
        beam.setPos(x, y, gz + 2.3)

        def fn(node, t):
            f = ease(min(1, t * 1.25))
            rock.setPos(x, y, gz + (1 - f) * 2.1)
            beam.setAlphaScale(1 - t)
        self.add_effect(root, 0.75, fn)

    def add_launch(self):
        root = self.dyn.attachNewNode('launch')
        start = Vec3(self.ufo_pos)
        bolt = self.card(root, 0.6, (1, 0.6, 0.2, 1))

        def fn(node, t):
            bolt.setPos(start + Vec3(1.5, 2.0, 6) * t)
            node.setAlphaScale(1 - t)
        self.add_effect(root, 0.6, fn)

    # ------------------------------------------------------------ alien hover + end
    def update_hover(self):
        self.hover_np.getChildren().detach()
        s = self.state
        if self.mode != 'alien' or s.turn != ALIEN or s.result:
            return
        p = self.pick()
        if p is None:
            return
        a = ('boulder', p) if s.blk.get(p, OPEN) == OPEN else ('debris', p)
        ok = s.legal_alien(a)
        x, y = self.world_xy(p)
        r = self.meshes.ring('hover', 0.4, 0.5, GREEN if ok else RED)
        self.meshes.additive(r)
        r.setPos(x, y, self.gz(x, y) + 0.06)
        r.reparentTo(self.hover_np)

    def show_end(self):
        s = self.state
        won = s.result == ROVER
        self.end_root = self.keep(self.base.aspect2d.attachNewNode('end'))
        self.panel(self.end_root, -0.75, 0.75, -0.3, 0.3, edge=CYAN if won else ORANGE)
        self.text(self.end_root, 'ROVER ESCAPED' if won else 'ROVER TRAPPED', (0, 0.13), 0.1,
                  CYAN if won else ORANGE, TextNode.ACenter)
        self.text(self.end_root, s.reason, (0, 0.03), 0.05, TEXT, TextNode.ACenter)
        self.text(self.end_root, f'Escape score {s.score:.0f}  ·  {s.tick} turns  ·  {s.shots} shots  ·  {s.hits} hits',
                  (0, -0.07), 0.042, MUTED, TextNode.ACenter)
        self.text(self.end_root, 'R replay  ·  N new map  ·  Esc menu', (0, -0.2), 0.045, TEXT, TextNode.ACenter)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Autonomous Escape v2 (3D)')
    ap.add_argument('--mode', choices=[m for m, _ in MODES], default='rover')
    ap.add_argument('--rover', choices=list(ROVER_BY_NAME), default='A*')
    ap.add_argument('--alien', choices=list(ALIEN_BY_NAME), default='Core')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--speed', type=int, default=2)
    ap.add_argument('--camera', choices=CAMERAS)
    ap.add_argument('--play', action='store_true', help='skip the menu')
    args = ap.parse_args(argv)
    configure()
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
    game = Game(base, args.mode, args.rover, args.alien, args.seed, args.speed, args.camera)
    if args.play:
        game.start_game()
    base.run()
