"""Autonomous Escape: pygame-ce front end.

Modes
  AI vs AI          pick any rover and alien and watch them play
  You are the Alien click tiles to strike; the AI rover drives through the fog
  You are the Rover arrow keys / WASD to drive (bump a soft boulder to shoot)

The board is a procedurally lit Phobos surface (ui/terrain.py): the maze walls
are rubble ridges, the corridors are groove floors. The rover knows the ridges
from the orbital map, so the whole maze is always visible; the fog only hides
boulders outside its radar. Debris streaks are visible anywhere in the sky.
"""
import math
import os
import random

import pygame as pg

from escape import GameState, OPEN, SOFT, HARD, ROVER, ALIEN, INF
from escape.agents import ROVER_BY_NAME, ALIEN_BY_NAME, make_rover, make_alien, CoreAlien
from escape.graph import astar, manhattan
from . import sprites, terrain

W, H = 1280, 780
TILE = 36
GX, GY = 12, 12
PX = 792                      # panel left edge
PW = W - PX - 16
ASSETS = os.path.join(os.path.dirname(__file__), 'assets')

# palette (matches the deck: deep space purple, cyan rover, orange alien)
BG = (10, 9, 24)
PANEL = (20, 18, 42, 222)
PANEL_EDGE = (58, 52, 108)
TEXT = (236, 236, 246)
MUTED = (156, 154, 184)
DIM = (100, 98, 128)
CYAN = (86, 220, 232)
ORANGE = (255, 158, 64)
RED = (240, 84, 96)
GREEN = (110, 220, 140)
YELLOW = (250, 210, 90)
PURPLE = (170, 130, 255)
FOG = (6, 8, 22)

MODES = [('ai', 'AI vs AI'), ('alien', 'You are the Alien'), ('rover', 'You are the Rover')]
SPEEDS = [1, 2, 4, 8, 16]
DIRKEYS = {pg.K_UP: (-1, 0), pg.K_w: (-1, 0), pg.K_DOWN: (1, 0), pg.K_s: (1, 0),
           pg.K_LEFT: (0, -1), pg.K_a: (0, -1), pg.K_RIGHT: (0, 1), pg.K_d: (0, 1)}
COMET_DIR = (0.58, -0.81)     # debris streaks come in from the upper right


def smooth(pts, rounds=3):
    """Chaikin corner cutting: the grid route drawn as a flowing curve."""
    for _ in range(rounds):
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            out += [(0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1), (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)]
        out.append(pts[-1])
        pts = out
    return pts


def beads(pts, gap):
    """Evenly spaced points along a polyline."""
    out, carry = [], gap
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        while carry <= seg:
            t = carry / seg
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            carry += gap
        carry -= seg
    return out


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


class App:
    def __init__(self, mode='ai', rover='A*', alien='Core', seed=None, speed=4, headless=False):
        pg.init()
        pg.display.set_caption('Autonomous Escape')
        flags = 0 if os.environ.get('SDL_VIDEODRIVER') == 'dummy' else pg.SCALED | pg.RESIZABLE
        self.screen = pg.display.set_mode((W, H), flags)
        self.clock = pg.time.Clock()
        self.fonts = {}
        self.text_cache = {}
        self.mode = mode
        self.rover_name, self.alien_name = rover, alien
        self.seed = seed if seed is not None else random.randrange(1000)
        self.speed = speed
        self.screen_name = 'menu'
        self.menu_row = 0
        self.show_fog = True
        self.show_plan = True
        self.show_cut = True
        self.show_grid = False
        self.paused = False
        self.headless = headless
        self.running = True
        self.terrain_cache = {}
        self.boulders = {}
        self.rover_img = sprites.rover(46)
        self.ufo_img = sprites.ufo(66)
        self.base_img = sprites.base(TILE)
        self.glow_cyan = sprites.glow(int(TILE * 1.6), CYAN, 70)
        self.glow_orange = sprites.glow(14, (255, 170, 70), 200)
        self.glow_red = sprites.glow(30, RED, 160)
        self.space = self.make_space()
        self.shadow_small = self.shadow(42, 30, 90)
        self.shadow_big = self.shadow(46, 15, 70)

    # ------------------------------------------------------------ helpers
    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pg.font.SysFont('segoeui,arial,helvetica', size, bold=bold)
        return self.fonts[key]

    def text(self, s, size=16, color=TEXT, bold=False):
        key = (s, size, color, bold)
        surf = self.text_cache.get(key)
        if surf is None:
            surf = self.font(size, bold).render(s, True, color)
            if len(self.text_cache) > 800:
                self.text_cache.clear()
            self.text_cache[key] = surf
        return surf

    def blit_text(self, s, pos, size=16, color=TEXT, bold=False, anchor='topleft', target=None):
        surf = self.text(s, size, color, bold)
        rect = surf.get_rect(**{anchor: pos})
        (target or self.screen).blit(surf, rect)
        return rect

    def wrap(self, s, size, width):
        words, lines, cur = s.split(), [], ''
        for w in words:
            t = (cur + ' ' + w).strip()
            if self.font(size).size(t)[0] <= width:
                cur = t
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    @staticmethod
    def tile_rect(p):
        return pg.Rect(GX + p[1] * TILE, GY + p[0] * TILE, TILE, TILE)

    @staticmethod
    def tile_center(p):
        return (GX + p[1] * TILE + TILE / 2, GY + p[0] * TILE + TILE / 2)

    @staticmethod
    def local(p):
        """Tile centre in board-surface coordinates."""
        return (p[1] * TILE + TILE / 2, p[0] * TILE + TILE / 2)

    def tile_at(self, pos):
        x, y = pos
        c, r = (x - GX) // TILE, (y - GY) // TILE
        n = self.state.map.size if hasattr(self, 'state') else 0
        if 0 <= r < n and 0 <= c < n:
            return (int(r), int(c))
        return None

    def boulder_img(self, p, kind):
        key = (kind, (p[0] * 7 + p[1] * 13) % 6)
        if key not in self.boulders:
            self.boulders[key] = sprites.boulder(kind, key[1], TILE)
        return self.boulders[key]

    @staticmethod
    def shadow(w, h, alpha):
        s = pg.Surface((w, h), pg.SRCALPHA)
        pg.draw.ellipse(s, (0, 0, 0, alpha), s.get_rect())
        return s

    def make_space(self):
        path = os.path.join(ASSETS, 'space.jpg')
        try:
            img = pg.image.load(path).convert()
            img = pg.transform.smoothscale(img, (W, H))
        except (pg.error, FileNotFoundError):
            img = pg.Surface((W, H))
            img.fill(BG)
            rng = random.Random(5)
            for _ in range(260):
                c = rng.randint(70, 220)
                img.set_at((rng.randrange(W), rng.randrange(H)), (c, c, min(255, c + 30)))
        dark = pg.Surface((W, H), pg.SRCALPHA)
        dark.fill((6, 5, 18, 120))
        img.blit(dark, (0, 0))
        return img

    def card(self, x, y, w, h, edge=PANEL_EDGE):
        key = ('card', w, h, edge)
        surf = self.text_cache.get(key)
        if surf is None:
            surf = pg.Surface((w, h), pg.SRCALPHA)
            pg.draw.rect(surf, PANEL, surf.get_rect(), border_radius=10)
            pg.draw.rect(surf, edge, surf.get_rect(), 1, border_radius=10)
            self.text_cache[key] = surf
        self.screen.blit(surf, (x, y))
        return pg.Rect(x, y, w, h)

    # ------------------------------------------------------------ game setup
    def start_game(self, new_seed=False):
        if new_seed:
            self.seed = random.randrange(1000)
        if self.seed not in self.terrain_cache and not self.headless:
            self.screen.blit(self.space, (0, 0))
            self.blit_text('Landing on Phobos...', (W // 2, H // 2), 28, TEXT, True, 'center')
            pg.display.flip()
        self.state = GameState.new(self.seed)
        self.rover_ai = make_rover(self.rover_name, self.seed) if self.mode != 'rover' else None
        self.alien_ai = make_alien(self.alien_name, self.seed, **self._alien_kw()) if self.mode != 'alien' else None
        for a in (self.rover_ai, self.alien_ai):
            if a:
                a.reset()
        self.core_view = CoreAlien()
        if self.seed not in self.terrain_cache:
            if len(self.terrain_cache) > 3:
                self.terrain_cache.clear()
            self.terrain_cache[self.seed] = terrain.build(self.state.map, TILE)
        S = self.state.map.size * TILE
        self.ground = self.make_ground()
        self.layer = pg.Surface((S, S), pg.SRCALPHA)
        self.fog_sheets = {}
        self.effects = []
        self.rover_from = self.state.rover
        self.rover_t0 = 0
        self.rover_angle = self.rover_angle_to = 0.0
        self.last_tick = pg.time.get_ticks()
        self.ufo_pos = [S * 0.5, S * 0.2]
        self.ufo_goal = None
        self.ufo_until = 0
        self.next_time = pg.time.get_ticks() + 700
        self.paused = False
        self.step_once = False
        self.flash = ''
        self.flash_t = 0
        self.overlay_cache = None
        self.screen_name = 'game'
        pg.key.set_repeat(260, 150)

    def make_ground(self):
        """Terrain plus the static markings: no-strike zones and the base."""
        m = self.state.map
        ground = self.terrain_cache[self.seed].copy()
        if self.mode == 'alien':      # show where strikes are not allowed
            marks = pg.Surface(ground.get_size(), pg.SRCALPHA)
            R = m.protect
            for centre in (m.start, m.goal):
                box = pg.Rect((centre[1] - R) * TILE, (centre[0] - R) * TILE, (2 * R + 1) * TILE, (2 * R + 1) * TILE)
                pg.draw.rect(marks, (*CYAN, 22), box, border_radius=TILE)
                pg.draw.rect(marks, (*CYAN, 120), box, 2, border_radius=TILE)
            ground.blit(marks, (0, 0))
        ground.blit(self.base_img, self.base_img.get_rect(center=self.local(m.start)))
        return ground

    def _alien_kw(self):
        return {'time_limit': 0.5} if self.alien_name == 'MCTS' else {}

    # ------------------------------------------------------------ actions
    def do_rover(self, action):
        s = self.state
        landing = [p for p, t in s.pending.items() if t == 1]
        hits, fuel = s.hits, s.fuel
        self.rover_from = s.rover
        self.rover_t0 = pg.time.get_ticks()
        s.apply_rover(action)
        now = pg.time.get_ticks()
        self.last_tick = now
        if action[0] == 'move' and s.rover != self.rover_from:
            dr, dc = s.rover[0] - self.rover_from[0], s.rover[1] - self.rover_from[1]
            self.rover_angle_to = math.degrees(math.atan2(-dr, dc))
            self.stamp_tracks(self.rover_from, s.rover)
            self.effects.append(('dust', (self.rover_from, s.rover), now, 650))
        if action[0] == 'shoot':
            dr, dc = action[1][0] - s.rover[0], action[1][1] - s.rover[1]
            self.rover_angle_to = math.degrees(math.atan2(-dr, dc))
            self.effects.append(('laser', (s.rover, action[1]), now, 380))
            self.effects.append(('float', (action[1], f'-{s.rules.shot_cost}', CYAN), now, 900))
        for p in landing:
            self.effects.append(('impact', p, now, 700))
        if s.hits > hits:
            self.effects.append(('hit', s.rover, now, 700))
            self.effects.append(('float', (s.rover, f'-{s.rules.debris_damage} FUEL', RED), now, 1200))
        self.overlay_cache = None
        if s.turn == ALIEN and self.mode == 'rover':
            self.next_time = now + 450

    def do_alien(self, action):
        s = self.state
        s.apply_alien(action)
        now = pg.time.get_ticks()
        if action:
            kind, p = action
            self.effects.append(('drop' if kind == 'boulder' else 'launch', p, now, 650))
            if self.viewer() == 'god' or s.in_radar(p):
                x, y = self.local(p)
                self.ufo_goal = (x, y - 40)
                self.ufo_until = now + 900
        self.overlay_cache = None
        self.next_time = now + self.interval()

    def interval(self):
        return int(1000 / self.speed)

    def stamp_tracks(self, a, b):
        """Press two wheel ruts into the ground between tiles a and b."""
        (x0, y0), (x1, y1) = self.local(a), self.local(b)
        ox, oy = min(x0, x1) - 10, min(y0, y1) - 10
        stamp = pg.Surface((abs(x1 - x0) + 20, abs(y1 - y0) + 20), pg.SRCALPHA)
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1
        nx, ny = -dy / L * 6, dx / L * 6
        for sgn in (-1, 1):
            pg.draw.line(stamp, (38, 32, 28, 80), (x0 - ox + sgn * nx, y0 - oy + sgn * ny),
                         (x1 - ox + sgn * nx, y1 - oy + sgn * ny), 3)
        self.ground.blit(stamp, (ox, oy))

    def human_rover(self, d):
        s = self.state
        if s.result or s.turn != ROVER:
            return
        tgt = (s.rover[0] + d[0], s.rover[1] + d[1])
        if not s.map.is_open(tgt):
            self.say('Ridge wall: no way through')
            return
        b = s.blk.get(tgt, OPEN)
        if b == HARD:
            self.say('Fused boulder: the laser cannot break it')
            return
        self.do_rover(('shoot', tgt) if b == SOFT else ('move', tgt))

    def human_alien_action(self, p, button):
        s = self.state
        if p is None or s.result or s.turn != ALIEN:
            return None
        if button == 3:
            return ('debris', p)
        return ('boulder', p) if s.blk.get(p, OPEN) == OPEN else ('debris', p)

    def say(self, msg):
        self.flash, self.flash_t = msg, pg.time.get_ticks()

    # ------------------------------------------------------------ loop
    def run(self):
        while self.running:
            self.clock.tick(60)
            for e in pg.event.get():
                self.handle(e)
            self.update()
            self.draw()
            pg.display.flip()
        pg.quit()

    def handle(self, e):
        if e.type == pg.QUIT:
            self.running = False
        elif self.screen_name == 'menu':
            self.handle_menu(e)
        else:
            self.handle_game(e)

    def update(self):
        if self.screen_name != 'game' or self.state.result:
            return
        s, now = self.state, pg.time.get_ticks()
        if self.mode == 'ai':
            if (self.paused and not self.step_once) or (not self.step_once and now < self.next_time):
                return
            self.step_once = False
            if s.turn == ALIEN:
                self.do_alien(self.alien_ai.act(s))
            else:
                self.do_rover(self.rover_ai.act(s.rover_view()))
            self.next_time = pg.time.get_ticks() + self.interval()
        elif self.mode == 'alien':
            if s.turn == ROVER and not self.paused and now >= self.next_time:
                self.do_rover(self.rover_ai.act(s.rover_view()))
                self.next_time = pg.time.get_ticks() + self.interval()
        elif self.mode == 'rover':
            if s.turn == ALIEN and now >= self.next_time:
                self.do_alien(self.alien_ai.act(s))

    # ------------------------------------------------------------ menu
    def menu_rows(self):
        return [
            ('Mode', dict(MODES)[self.mode], True),
            ('Rover AI', self.rover_name if self.mode != 'rover' else 'You', self.mode != 'rover'),
            ('Alien AI', self.alien_name if self.mode != 'alien' else 'You', self.mode != 'alien'),
            ('Map', f'#{self.seed}', True),
            ('Speed', f'{self.speed} moves / s', self.mode != 'rover'),
        ]

    def menu_change(self, row, d):
        if row == 0:
            keys = [m for m, _ in MODES]
            self.mode = keys[(keys.index(self.mode) + d) % len(keys)]
        elif row == 1 and self.mode != 'rover':
            names = list(ROVER_BY_NAME)
            self.rover_name = names[(names.index(self.rover_name) + d) % len(names)]
        elif row == 2 and self.mode != 'alien':
            names = list(ALIEN_BY_NAME)
            self.alien_name = names[(names.index(self.alien_name) + d) % len(names)]
        elif row == 3:
            self.seed = (self.seed + d) % 1000
        elif row == 4:
            i = SPEEDS.index(self.speed) if self.speed in SPEEDS else 2
            self.speed = SPEEDS[(i + d) % len(SPEEDS)]

    def menu_layout(self):
        rows = []
        for i in range(5):
            y = 262 + i * 60
            rows.append((pg.Rect(90, y, 540, 50), pg.Rect(390, y + 7, 36, 36), pg.Rect(586, y + 7, 36, 36)))
        start = pg.Rect(90, 262 + 5 * 60 + 14, 540, 58)
        return rows, start

    def handle_menu(self, e):
        rows, start = self.menu_layout()
        if e.type == pg.KEYDOWN:
            if e.key in (pg.K_UP, pg.K_w):
                self.menu_row = (self.menu_row - 1) % 5
            elif e.key in (pg.K_DOWN, pg.K_s):
                self.menu_row = (self.menu_row + 1) % 5
            elif e.key in (pg.K_LEFT, pg.K_a):
                self.menu_change(self.menu_row, -1)
            elif e.key in (pg.K_RIGHT, pg.K_d):
                self.menu_change(self.menu_row, 1)
            elif e.key == pg.K_r:
                self.seed = random.randrange(1000)
            elif e.key in (pg.K_RETURN, pg.K_SPACE):
                self.start_game()
            elif e.key == pg.K_ESCAPE:
                self.running = False
        elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
            if start.collidepoint(e.pos):
                self.start_game()
            for i, (row, left, right) in enumerate(rows):
                if left.collidepoint(e.pos):
                    self.menu_row = i
                    self.menu_change(i, -1)
                elif right.collidepoint(e.pos) or row.collidepoint(e.pos):
                    self.menu_row = i
                    self.menu_change(i, 1)

    def draw_menu(self):
        sc = self.screen
        now = pg.time.get_ticks()
        sc.blit(self.space, (0, 0))
        self.blit_text('AUTONOMOUS ESCAPE', (88, 92), 66, TEXT, True)
        self.blit_text('An asteroid has hit the mining base on Phobos. Drive the last rover across the',
                       (92, 180), 19, MUTED)
        self.blit_text('grooved debris field to the launch pad before the alien seals every route.',
                       (92, 206), 19, MUTED)
        rows, start = self.menu_layout()
        for i, ((label, value, enabled), (row, left, right)) in enumerate(zip(self.menu_rows(), rows)):
            sel = i == self.menu_row
            self.card(row.x, row.y, row.w, row.h, CYAN if sel else PANEL_EDGE)
            self.blit_text(label, (row.x + 22, row.centery), 20, TEXT if enabled else DIM, True, 'midleft')
            if enabled:
                for b, ch in ((left, '<'), (right, '>')):
                    pg.draw.rect(sc, (44, 40, 86), b, border_radius=8)
                    self.blit_text(ch, b.center, 20, CYAN, True, 'center')
            self.blit_text(value, ((left.right + right.left) // 2, row.centery), 20,
                           TEXT if enabled else DIM, False, 'center')
        pulse = 0.5 + 0.5 * math.sin(now / 400)
        pg.draw.rect(sc, (int(60 + 26 * pulse), int(190 + 30 * pulse), int(206 + 26 * pulse)), start, border_radius=12)
        self.blit_text('LAUNCH', start.center, 24, BG, True, 'center')
        self.blit_text('Arrows choose, Enter starts, R random map, Esc quits', (92, start.bottom + 16), 16, MUTED)
        # rules card
        from escape import DEFAULT as R
        card = self.card(690, 262, 548, 448)
        self.blit_text('How it plays', (card.x + 24, card.y + 18), 22, TEXT, True)
        sc.blit(self.rover_img, self.rover_img.get_rect(center=(card.right - 92, card.y + 34)))
        sc.blit(self.ufo_img, self.ufo_img.get_rect(center=(card.right - 44, card.y + 30 + 4 * math.sin(now / 300))))
        lines = [
            (CYAN, f'The rover acts every turn. A move or a wait costs {R.move_cost} fuel.'),
            (CYAN, f'Laser: blast an adjacent loose boulder for {R.shot_cost} fuel.'),
            (CYAN, f'The ridges are known from orbit. Radar sees boulders {R.radar} tiles around.'),
            (ORANGE, f'The alien strikes every {R.alien_every} turns, {R.alien_budget} strikes in all.'),
            (ORANGE, 'Boulder: blocks a tile at once. Hidden from the rover outside its radar.'),
            (ORANGE, f'Debris: a glowing streak, visible anywhere, lands {R.debris_delay} turns later. It fuses a '
                     f'boulder (laser-proof) or hits the rover for {R.debris_damage} fuel.'),
            (MUTED, 'No strikes next to the rover or near the base and the pad.'),
            (GREEN, f'Start with {R.fuel_mult}x the fuel of the shortest route. Reach the pad to win.'),
        ]
        y = card.y + 64
        for col, line in lines:
            pg.draw.circle(sc, col, (card.x + 30, y + 11), 5)
            for part in self.wrap(line, 17, card.width - 70):
                self.blit_text(part, (card.x + 46, y), 17, TEXT)
                y += 23
            y += 8

    # ------------------------------------------------------------ game input
    def handle_game(self, e):
        s = self.state
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_ESCAPE or e.key == pg.K_m:
                self.screen_name = 'menu'
                pg.key.set_repeat()
            elif e.key == pg.K_r:
                self.start_game()
            elif e.key == pg.K_n:
                self.start_game(new_seed=True)
            elif e.key == pg.K_f:
                self.show_fog = not self.show_fog
            elif e.key == pg.K_p:
                self.show_plan = not self.show_plan
            elif e.key == pg.K_c:
                self.show_cut = not self.show_cut
            elif e.key == pg.K_g:
                self.show_grid = not self.show_grid
            elif e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self.speed = SPEEDS[min(len(SPEEDS) - 1, SPEEDS.index(self.speed) + 1)] if self.speed in SPEEDS else 4
            elif e.key in (pg.K_MINUS, pg.K_KP_MINUS):
                self.speed = SPEEDS[max(0, SPEEDS.index(self.speed) - 1)] if self.speed in SPEEDS else 4
            elif self.mode == 'rover':
                if e.key in DIRKEYS:
                    self.human_rover(DIRKEYS[e.key])
                elif e.key == pg.K_SPACE and s.turn == ROVER and not s.result:
                    self.do_rover(('wait', None))
            elif e.key == pg.K_SPACE:
                if self.mode == 'alien' and s.turn == ALIEN and not s.result:
                    self.do_alien(None)
                else:
                    self.paused = not self.paused
            elif e.key in (pg.K_PERIOD, pg.K_RIGHT) and self.mode == 'ai':
                self.paused = True
                self.step_once = True
        elif e.type == pg.MOUSEBUTTONDOWN and self.mode == 'alien' and e.button in (1, 3):
            a = self.human_alien_action(self.tile_at(e.pos), e.button)
            if a is None:
                return
            if s.legal_alien(a):
                self.do_alien(a)
            else:
                self.say(self.why_illegal(a[1]))

    def why_illegal(self, p):
        s = self.state
        if not s.map.is_open(p):
            return 'That is a ridge wall'
        if s.map.protected(p):
            return 'No-strike zone around the base and pad'
        if manhattan(p, s.rover) <= 1:
            return 'Too close to the rover'
        if p in s.pending:
            return 'Debris is already on its way there'
        if s.blk.get(p) == HARD:
            return 'Already fused'
        return 'Not allowed'

    # ------------------------------------------------------------ drawing
    def draw(self):
        if self.screen_name == 'menu':
            self.draw_menu()
        else:
            self.draw_game()

    def viewer(self):
        return 'rover' if self.mode == 'rover' else 'god'

    def overlays(self):
        """Rover's planned route (belief) and the alien's current min cut."""
        if self.overlay_cache is None:
            s = self.state
            v = s.rover_view()
            _, plan = astar(s.map, v.belief, s.rover, s.map.goal, s.rules)
            cut, need, seal, arrive = self.core_view.plan(s) if self.viewer() == 'god' else ([], 0, 0, 0)
            known, _ = astar(s.map, v.belief, s.rover, s.map.goal, s.rules)
            self.costs = (known, s.route_cost())
            self.overlay_cache = (plan, cut, need, seal, arrive)
        return self.overlay_cache

    def rover_xy(self, now):
        """Smoothly animated rover position in board coordinates."""
        dur = max(60, min(180, int(self.interval() * 0.8)))
        t = ease((now - self.rover_t0) / dur)
        (fx, fy), (tx, ty) = self.local(self.rover_from), self.local(self.state.rover)
        return lerp(fx, tx, t), lerp(fy, ty, t)

    def draw_game(self):
        sc = self.screen
        s = self.state
        now = pg.time.get_ticks()
        rover_view = self.viewer() == 'rover'
        plan, cut, need, seal, arrive = self.overlays()
        rx, ry = self.rover_xy(now)
        L = self.layer

        sc.blit(self.space, (0, 0))
        board = pg.Rect(GX, GY, s.map.size * TILE, s.map.size * TILE)
        sc.blit(self.ground, board)
        sc.set_clip(board)

        # boulders: truth, or the rover's memory in rover view
        blk = s.belief if rover_view else s.blk
        falling = {e[1] for e in self.effects if e[0] == 'drop' and now - e[2] < e[3]}
        for p, b in blk.items():
            if p in falling:
                continue
            img = self.boulder_img(p, b)
            if rover_view and not s.in_radar(p):
                img = img.copy()
                img.set_alpha(170)
            sc.blit(img, img.get_rect(center=self.tile_center(p)))
        self.draw_rover(rx + GX, ry + GY)

        # fog hides boulders outside the radar; the ridges stay visible
        if self.show_fog or rover_view:
            self.draw_fog(rx, ry, 140 if rover_view else 60, board)

        # everything the rover knows or can see in the sky
        L.fill((0, 0, 0, 0))
        self.draw_radar(L, rx, ry, now)
        if self.show_grid:
            for p in s.map.open_tiles:
                pg.draw.rect(L, (255, 255, 255, 22), pg.Rect(p[1] * TILE, p[0] * TILE, TILE, TILE), 1)
        if (now // 500) % 2:
            x, y = self.local(s.map.start)
            pg.draw.circle(L, RED, (x + TILE * 0.55, y - TILE * 0.55), 3)
        for p, ttl in s.pending.items():
            self.draw_reticle(L, p, ttl, now)
        if self.show_plan and len(plan) > 1 and not s.result:
            pts = smooth([(rx, ry)] + [self.local(p) for p in plan[1:]])
            pg.draw.lines(L, (*CYAN, 45), False, pts, 11)
            pg.draw.lines(L, (*CYAN, 230), False, pts, 3)
            for q in beads(pts, 15):
                pg.draw.circle(L, (240, 255, 255, 240), q, 2.6)
        self.draw_pad(L, s.map.goal, now)
        if self.show_cut and cut and not s.result:
            a = int(150 + 100 * (0.5 + 0.5 * math.sin(now / 220)))
            for p in cut:
                self.draw_brackets(L, p, (*ORANGE, a))
        if self.mode == 'alien' and s.turn == ALIEN and not s.result:
            self.draw_hover(L)
        self.draw_comets(L, now)
        self.draw_effects(L, now, rover_view)
        self.draw_ufo(L, now)
        sc.blit(L, board)
        self.draw_banner(plan, board)
        sc.set_clip(None)
        pg.draw.rect(sc, PANEL_EDGE, board.inflate(4, 4), 2, border_radius=3)
        self.draw_panel(now, plan, cut, seal, arrive)
        if s.result:
            self.draw_end(board)

    def draw_fog(self, rx, ry, alpha, board):
        """One big fog sheet with a soft hole in the middle, slid under the rover."""
        S = board.w
        sheet = self.fog_sheets.get(alpha)
        if sheet is None:
            sheet = pg.Surface((2 * S, 2 * S), pg.SRCALPHA)
            sheet.fill((*FOG, alpha))
            hole = self.make_hole((self.state.rules.radar + 0.5) * TILE, alpha)
            sheet.blit(hole, hole.get_rect(center=(S, S)), special_flags=pg.BLEND_RGBA_MIN)
            self.fog_sheets[alpha] = sheet
        self.screen.blit(sheet, board, pg.Rect(S - int(rx), S - int(ry), S, S))

    @staticmethod
    def make_hole(R, alpha):
        import numpy as np
        size = int(R * 2 + 40)
        surf = pg.Surface((size, size), pg.SRCALPHA)
        surf.fill((*FOG, alpha))
        yy, xx = np.mgrid[0:size, 0:size]
        d = np.hypot(xx - size / 2, yy - size / 2)
        t = np.clip((d - (R - 16)) / 22, 0, 1)
        a = pg.surfarray.pixels_alpha(surf)
        a[:] = (alpha * t * t * (3 - 2 * t)).T.astype(np.uint8)
        del a
        return surf

    def draw_radar(self, L, rx, ry, now):
        R = (self.state.rules.radar + 0.5) * TILE
        pg.draw.circle(L, (*CYAN, 16), (rx, ry), R)
        pg.draw.circle(L, (*CYAN, 120), (rx, ry), R, 2)
        a0 = now / 700
        pg.draw.line(L, (*CYAN, 120), (rx, ry), (rx + R * math.cos(a0), ry + R * math.sin(a0)), 2)
        step = 0.045
        for k in range(16):
            a1, a2 = a0 - k * step, a0 - (k + 1) * step
            pg.draw.polygon(L, (*CYAN, int(46 * (1 - k / 16))),
                            [(rx, ry), (rx + R * math.cos(a1), ry + R * math.sin(a1)),
                             (rx + R * math.cos(a2), ry + R * math.sin(a2))])

    def draw_pad(self, L, p, now):
        x, y = self.local(p)
        pulse = 0.5 + 0.5 * math.sin(now / 320)
        L.blit(self.glow_cyan, self.glow_cyan.get_rect(center=(x, y)))
        pg.draw.circle(L, (16, 30, 46, 240), (x, y), TILE * 0.62)
        pg.draw.circle(L, (*CYAN, 255), (x, y), TILE * 0.62, 3)
        pg.draw.circle(L, (230, 250, 255, 220), (x, y), TILE * 0.44, 2)
        R = TILE * (0.86 + 0.08 * pulse)
        rect = pg.Rect(0, 0, 2 * R, 2 * R)
        rect.center = (x, y)
        a = now / 900
        for k in range(4):
            pg.draw.arc(L, (*CYAN, 200), rect, a + k * math.pi / 2, a + k * math.pi / 2 + 0.9, 2)
        self.blit_text('H', (x, y + 1), 20, (230, 250, 255), True, 'center', target=L)

    def draw_reticle(self, L, p, ttl, now):
        x, y = self.local(p)
        pulse = 0.5 + 0.5 * math.sin(now / 150)
        col = (255, 96, 80, int(170 + 80 * pulse))
        pg.draw.circle(L, col, (x, y), 13 + 2 * pulse, 2)
        pg.draw.circle(L, col, (x, y), 5, 2)
        self.blit_text(str(ttl), (x + 13, y - 20), 14, (255, 140, 120), True, target=L)

    def draw_brackets(self, L, p, col):
        r = pg.Rect(p[1] * TILE, p[0] * TILE, TILE, TILE).inflate(-4, -4)
        k = 9
        for (cx, cy), (dx, dy) in ((r.topleft, (1, 1)), (r.topright, (-1, 1)),
                                   (r.bottomleft, (1, -1)), (r.bottomright, (-1, -1))):
            pg.draw.line(L, col, (cx, cy), (cx + dx * k, cy), 3)
            pg.draw.line(L, col, (cx, cy), (cx, cy + dy * k), 3)

    def comet_head(self, p, ttl, now):
        x, y = self.local(p)
        prog = min(1.0, (now - self.last_tick) / 700) * 0.7
        d = TILE * (0.6 + 2.2 * (ttl - prog))
        return x + COMET_DIR[0] * d, y + COMET_DIR[1] * d

    def draw_comets(self, L, now):
        for p, ttl in self.state.pending.items():
            hx, hy = self.comet_head(p, ttl, now)
            x, y = self.local(p)
            # dotted guide down to the landing spot
            n = int(math.hypot(hx - x, hy - y) // 9)
            for i in range(1, n):
                t = i / n
                pg.draw.circle(L, (255, 120, 100, 160), (lerp(hx, x, t), lerp(hy, y, t)), 1.5)
            for i in range(24, 0, -1):
                t = i / 24
                tx, ty = hx + COMET_DIR[0] * 56 * t, hy + COMET_DIR[1] * 56 * t
                pg.draw.circle(L, (255, int(140 + 80 * (1 - t)), 60, int(40 + 150 * (1 - t))), (tx, ty), 2 + 6 * (1 - t))
            L.blit(self.glow_orange, self.glow_orange.get_rect(center=(hx, hy)))
            pg.draw.circle(L, (255, 240, 200), (hx, hy), 4)

    def draw_rover(self, x, y):
        diff = (self.rover_angle_to - self.rover_angle + 180) % 360 - 180
        self.rover_angle += diff * 0.25
        img = pg.transform.rotate(self.rover_img, self.rover_angle)
        self.screen.blit(self.shadow_small, self.shadow_small.get_rect(center=(x + 4, y + 5)))
        self.screen.blit(img, img.get_rect(center=(x, y)))

    def draw_ufo(self, L, now):
        S = self.state.map.size * TILE
        if self.ufo_goal and now < self.ufo_until:
            gx, gy = self.ufo_goal
        else:
            gx = S * (0.5 + 0.34 * math.sin(now / 5200))
            gy = S * (0.42 + 0.3 * math.sin(now / 3900 + 1.3))
        self.ufo_pos[0] += (gx - self.ufo_pos[0]) * 0.12
        self.ufo_pos[1] += (gy - self.ufo_pos[1]) * 0.12
        x, y = self.ufo_pos
        y += 3 * math.sin(now / 260)
        L.blit(self.shadow_big, self.shadow_big.get_rect(center=(x + 22, y + 46)))
        L.blit(self.ufo_img, self.ufo_img.get_rect(center=(x, y)))

    def draw_hover(self, L):
        p = self.tile_at(pg.mouse.get_pos())
        if p is None:
            return
        s = self.state
        a = self.human_alien_action(p, 1)
        ok = a is not None and s.legal_alien(a)
        col = GREEN if ok else RED
        r = pg.Rect(p[1] * TILE, p[0] * TILE, TILE, TILE)
        pg.draw.rect(L, (*col, 255), r, 2, border_radius=4)
        if ok:
            label = 'Drop boulder' if a[0] == 'boulder' else 'Fuse with debris'
            tip = self.text(label + ('  (right-click: debris)' if a[0] == 'boulder' else ''), 14, TEXT)
            box = tip.get_rect(midbottom=(r.centerx, r.top - 6)).inflate(12, 6)
            box.clamp_ip(L.get_rect())
            pg.draw.rect(L, (20, 18, 40, 230), box, border_radius=6)
            L.blit(tip, tip.get_rect(center=box.center))

    def draw_effects(self, L, now, rover_view=False):
        keep = []
        for kind, data, t0, dur in self.effects:
            t = (now - t0) / dur
            if t >= 1:
                continue
            keep.append((kind, data, t0, dur))
            if rover_view and kind in ('drop', 'launch') and not self.state.in_radar(data):
                continue
            if kind == 'laser':
                a, b = self.local(data[0]), self.local(data[1])
                fade = int(255 * (1 - t))
                pg.draw.line(L, (255, 60, 110, fade // 3), a, b, 10)
                pg.draw.line(L, (255, 90, 130, fade), a, b, 4)
                pg.draw.line(L, (255, 230, 240, fade), a, b, 2)
                for k in range(8):
                    ang = k * math.pi / 4 + 0.4
                    d = 6 + 22 * t
                    pg.draw.circle(L, (255, 200, 120, fade), (b[0] + d * math.cos(ang), b[1] + d * math.sin(ang)), 2)
            elif kind == 'impact':
                x, y = self.local(data)
                fade = int(255 * (1 - t))
                if t < 0.35:
                    g = pg.transform.smoothscale(self.glow_orange, (int(30 + 60 * t), int(30 + 60 * t)))
                    L.blit(g, g.get_rect(center=(x, y)))
                pg.draw.circle(L, (200, 180, 160, fade // 2), (x, y), 8 + 26 * t, 4)
                for k in range(10):
                    ang = k * 0.63 + 0.2
                    d = 4 + 30 * t
                    pg.draw.circle(L, (170, 150, 130, fade), (x + d * math.cos(ang), y + d * math.sin(ang) * 0.8), 2)
            elif kind == 'hit':
                x, y = self.local(data)
                g = self.glow_red.copy()
                g.set_alpha(int(255 * (1 - t)))
                L.blit(g, g.get_rect(center=(x, y)))
            elif kind == 'drop':
                x, y = self.local(data)
                img = self.boulder_img(data, SOFT)
                fall = (1 - ease(t * 1.3)) * 70
                pg.draw.ellipse(L, (0, 0, 0, int(60 + 60 * t)), pg.Rect(x - 12, y - 4, 24, 10))
                if t < 0.75:
                    pg.draw.polygon(L, (255, 230, 120, 50), [(x - 8, y - 70), (x + 8, y - 70), (x + 16, y), (x - 16, y)])
                L.blit(img, img.get_rect(center=(x, y - fall)))
                if t > 0.75:
                    pg.draw.circle(L, (200, 180, 160, int(400 * (1 - t))), (x, y), 10 + 40 * (t - 0.75), 3)
            elif kind == 'launch':
                x, y = self.local(data)
                ux, uy = self.ufo_pos
                fade = int(255 * (1 - t))
                y0 = uy - 14 - 300 * t
                pg.draw.line(L, (255, 160, 60, fade), (ux + 0.12 * (uy - y0), y0), (ux + 0.12 * (uy - y0 + 34), y0 - 34), 4)
            elif kind == 'dust':
                (x0, y0), (x1, y1) = self.local(data[0]), self.local(data[1])
                for k in range(5):
                    f = k / 5
                    x, y = lerp(x0, x1, 0.2 + 0.5 * f), lerp(y0, y1, 0.2 + 0.5 * f)
                    rad = 4 + 10 * t + 2 * k
                    pg.draw.circle(L, (190, 170, 150, int(70 * (1 - t) * (1 - f * 0.5))),
                                   (x + 6 * math.sin(k * 2.1), y + 6 * math.cos(k * 1.7)), rad)
            elif kind == 'float':
                p, msg, col = data
                x, y = self.local(p)
                surf = self.text(msg, 16, col, True).copy()
                surf.set_alpha(int(255 * (1 - t)))
                L.blit(surf, surf.get_rect(center=(x, y - 18 - 28 * t)))
        self.effects = keep

    def draw_banner(self, plan, board):
        s = self.state
        if s.result:
            return
        route = set(plan[1:8])
        soon = [p for p, t in s.pending.items() if p in route or p == s.rover]
        if not soon:
            return
        here = s.rover in s.pending
        msg = 'Debris landing on the rover next turn: move!' if here else 'Debris incoming on the planned route'
        tip = self.text(msg, 16, (255, 230, 210), True)
        box = tip.get_rect(midtop=(board.centerx, board.y + 10)).inflate(22, 10)
        surf = pg.Surface(box.size, pg.SRCALPHA)
        pg.draw.rect(surf, (120, 40, 30, 220), surf.get_rect(), border_radius=8)
        self.screen.blit(surf, box)
        self.screen.blit(tip, tip.get_rect(center=box.center))

    # ------------------------------------------------------------ panel
    def draw_panel(self, now, plan, cut, seal, arrive):
        s = self.state
        sc = self.screen
        rover_view = self.viewer() == 'rover'
        self.blit_text('AUTONOMOUS ESCAPE', (PX, 10), 26, TEXT, True)
        self.blit_text(f'{dict(MODES)[self.mode]}   ·   Map #{s.map.seed}', (PX + 2, 46), 16, MUTED)

        # matchup
        r = self.card(PX, 76, PW, 58)
        rv = self.rover_ai.name if self.rover_ai else 'You'
        al = self.alien_ai.name if self.alien_ai else 'You'
        self.blit_text('ROVER', (r.x + 16, r.y + 8), 13, CYAN, True)
        self.blit_text(rv, (r.x + 16, r.y + 25), 20, TEXT, True)
        self.blit_text('ALIEN', (r.centerx + 16, r.y + 8), 13, ORANGE, True)
        self.blit_text(al, (r.centerx + 16, r.y + 25), 20, TEXT, True)
        self.blit_text('vs', (r.centerx - 8, r.centery), 15, DIM, False, 'center')

        # fuel
        r = self.card(PX, 142, PW, 86)
        self.blit_text('FUEL', (r.x + 16, r.y + 9), 13, CYAN, True)
        self.blit_text(f'{max(s.fuel, 0)} / {s.fuel0}', (r.right - 16, r.y + 6), 18, TEXT, True, 'topright')
        bar = pg.Rect(r.x + 16, r.y + 33, r.width - 32, 14)
        pg.draw.rect(sc, (40, 38, 70), bar, border_radius=7)
        frac = max(0, s.fuel) / s.fuel0
        col = GREEN if frac > 0.45 else YELLOW if frac > 0.2 else RED
        pg.draw.rect(sc, col, (bar.x, bar.y, int(bar.width * frac), bar.height), border_radius=7)
        if rover_view:
            cost, label = self.costs[0], 'Best known route'
        else:
            cost, label = self.costs[1], 'Cheapest true route'
        cost_txt = 'none' if cost == INF else f'{cost:.0f} fuel'
        ok = cost != INF and cost <= s.fuel
        if cost != INF and 0 < bar.width * cost / s.fuel0 <= bar.width:
            mx = bar.x + int(bar.width * min(cost, s.fuel0) / s.fuel0)
            pg.draw.line(sc, TEXT, (mx, bar.y - 4), (mx, bar.bottom + 4), 2)
        self.blit_text(f'{label}: {cost_txt}', (r.x + 16, r.y + 56), 15, TEXT if ok else RED)
        self.blit_text(f'shots {s.shots}  ·  hits {s.hits}', (r.right - 16, r.y + 56), 15, MUTED, False, 'topright')

        # alien clock
        r = self.card(PX, 236, PW, 86)
        self.blit_text('ALIEN', (r.x + 16, r.y + 9), 13, ORANGE, True)
        self.blit_text(f'Turn {s.tick}', (r.right - 16, r.y + 6), 18, TEXT, True, 'topright')
        for i in range(s.rules.alien_budget):
            cx = r.x + 22 + i * 22
            used = i >= s.budget
            pg.draw.circle(sc, (70, 60, 60) if used else ORANGE, (cx, r.y + 40), 7, 2 if used else 0)
        if s.result:
            msg = ''
        elif s.turn == ALIEN:
            msg = 'YOUR STRIKE: click a tile (Space to hold)' if self.mode == 'alien' else 'Alien is striking...'
        elif s.budget == 0:
            msg = 'Alien is out of strikes'
        else:
            k = s.rules.alien_every - s.tick % s.rules.alien_every
            msg = f'Next strike in {k} turn{"s" if k > 1 else ""}'
        if s.pending and not s.result:
            msg += f'   ·   {len(s.pending)} debris incoming'
        self.blit_text(msg, (r.x + 16, r.y + 58), 15, ORANGE if s.turn == ALIEN else TEXT, s.turn == ALIEN)

        # AI thoughts
        r = self.card(PX, 330, PW, 70)
        self.blit_text('WHAT THE AI SEES', (r.x + 16, r.y + 9), 13, PURPLE, True)
        lines = []
        if self.rover_ai:
            route = f'route {len(plan) - 1} tiles' if len(plan) > 1 else 'no known route'
            lines.append((CYAN, f'{self.rover_ai.name}: ' + (self.rover_ai.info or route)))
        if self.alien_ai and rover_view:
            lines.append((ORANGE, f'{self.alien_ai.name}: plans hidden by the fog'))
        elif self.alien_ai:
            lines.append((ORANGE, f'{self.alien_ai.name}: ' + (self.alien_ai.info or 'waiting')))
        elif cut:
            lines.append((ORANGE, f'Min cut: {len(cut)} tiles, seal {seal} turns vs rover {arrive}'))
        for i, (col, line) in enumerate(lines[:2]):
            t = line if self.font(15).size(line)[0] < r.width - 32 else line[:60] + '...'
            self.blit_text(t, (r.x + 16, r.y + 28 + i * 20), 15, col)

        # event log
        r = self.card(PX, 408, PW, 166)
        self.blit_text('MISSION LOG', (r.x + 16, r.y + 9), 13, MUTED, True)
        events = s.log
        if rover_view:
            events = [(t, m) for t, m in s.log if 'drops a boulder' not in m]
        for i, (t, m) in enumerate(reversed(events[-6:])):
            self.blit_text(f'T{t:>3}', (r.x + 16, r.y + 32 + i * 21), 14, DIM)
            self.blit_text(m, (r.x + 62, r.y + 32 + i * 21), 14, TEXT if i == 0 else MUTED)

        # controls + legend
        r = self.card(PX, 582, PW, 186)
        self.blit_text('CONTROLS', (r.x + 16, r.y + 9), 13, MUTED, True)
        if self.mode == 'rover':
            ctl = ['Arrows / WASD  drive (bump a loose boulder to shoot)', 'Space  wait one turn']
        elif self.mode == 'alien':
            ctl = ['Left-click  boulder on open ground, debris on a boulder',
                   'Right-click  debris on open ground', 'Space  hold this strike']
        else:
            ctl = [f'Space  pause{" (paused)" if self.paused else ""}   ·   →  step one move',
                   f'+ / -  speed ({self.speed} moves/s)']
        ctl += ['F fog · P route · C cut · G grid · R replay · N new map · Esc']
        for i, line in enumerate(ctl):
            self.blit_text(line, (r.x + 16, r.y + 32 + i * 22), 14, TEXT)
        ly = r.bottom - 34
        lx = r.x + 12
        for key, label in (('soft', 'loose boulder'), ('hard', 'fused'), ('debris', 'incoming'),
                           ('cut', 'min cut'), ('pad', 'pad')):
            if key in ('soft', 'hard'):
                img = pg.transform.smoothscale(self.boulder_img((0, 0), SOFT if key == 'soft' else HARD), (30, 30))
                sc.blit(img, img.get_rect(center=(lx + 10, ly + 10)))
            elif key == 'debris':
                pg.draw.circle(sc, (255, 96, 80), (lx + 10, ly + 10), 8, 2)
                pg.draw.circle(sc, (255, 96, 80), (lx + 10, ly + 10), 3, 2)
            elif key == 'cut':
                pg.draw.rect(sc, ORANGE, (lx + 2, ly + 2, 16, 16), 2, border_radius=3)
            else:
                pg.draw.circle(sc, CYAN, (lx + 10, ly + 10), 8, 2)
            t = self.blit_text(label, (lx + 24, ly + 1), 14, MUTED)
            lx = t.right + 12

        if self.flash and now - self.flash_t < 1800:
            tip = self.text(self.flash, 16, TEXT, True)
            box = tip.get_rect(midbottom=(GX + s.map.size * TILE // 2, GY + s.map.size * TILE - 14)).inflate(22, 10)
            pg.draw.rect(sc, (90, 28, 44), box, border_radius=8)
            sc.blit(tip, tip.get_rect(center=box.center))

    def draw_end(self, board):
        s = self.state
        shade = pg.Surface(board.size, pg.SRCALPHA)
        shade.fill((6, 6, 18, 150))
        self.screen.blit(shade, board)
        won = s.result == ROVER
        box = pg.Rect(0, 0, 540, 230)
        box.center = board.center
        self.card(box.x, box.y, box.w, box.h, CYAN if won else ORANGE)
        pg.draw.rect(self.screen, CYAN if won else ORANGE, box, 3, border_radius=10)
        title = 'ROVER ESCAPED' if won else 'ROVER TRAPPED'
        self.blit_text(title, (box.centerx, box.y + 26), 42, CYAN if won else ORANGE, True, 'midtop')
        self.blit_text(s.reason, (box.centerx, box.y + 90), 18, TEXT, False, 'midtop')
        detail = f'Escape score {s.score:.0f}   ·   {s.tick} turns   ·   {s.shots} shots   ·   {s.hits} hits'
        self.blit_text(detail, (box.centerx, box.y + 124), 16, MUTED, False, 'midtop')
        self.blit_text('R replay this map   ·   N new map   ·   Esc menu', (box.centerx, box.y + 176), 16, TEXT,
                       False, 'midtop')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Autonomous Escape')
    ap.add_argument('--mode', choices=[m for m, _ in MODES], default='ai')
    ap.add_argument('--rover', choices=list(ROVER_BY_NAME), default='A*')
    ap.add_argument('--alien', choices=list(ALIEN_BY_NAME), default='Core')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--speed', type=int, default=4)
    ap.add_argument('--play', action='store_true', help='skip the menu')
    args = ap.parse_args(argv)
    app = App(args.mode, args.rover, args.alien, args.seed, args.speed)
    if args.play:
        app.start_game()
    app.run()
