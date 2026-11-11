"""The cockpit HUD: compass tape, dashboard (fuel, radar scope, alien and
systems), radio messages, orbital minimap, alerts and the pilot reticle."""
import math

import numpy as np
from panda3d.core import LineSegs, NodePath, TextNode, TransparencyAttrib

from escape import ALIEN, INF, SOFT, HARD
from . import story
from .widgets import (CYAN, DIM, GREEN, MUTED, ORANGE, PURPLE, RED, TEXT, YELLOW, DynamicTexture, outline)

SPEAKER = {'CONTROL': CYAN, 'ROVER': TEXT, 'SIGNAL': ORANGE}


class Hud:
    def __init__(self, game):
        from direct.gui.DirectGui import DirectFrame
        from direct.gui.OnscreenImage import OnscreenImage
        self.g = game
        b = game.base
        ui = game.ui
        self.roots = []

        def root(parent, name):
            n = parent.attachNewNode(name)
            self.roots.append(n)
            return n
        s = game.state
        # top-left: mission
        tl = root(b.a2dTopLeft, 'tl')
        title = game.mission.title.upper() if game.mission else dict(game.MODES)[game.mode].upper()
        place = game.mission.place if game.mission else f'Map #{game.seed}'
        ui.text(tl, title, (0.06, -0.09), 0.055, TEXT, bold=True)
        rv = game.rover_ai.name if game.rover_ai else 'You'
        al = game.alien_ai.name if game.alien_ai else 'You'
        ui.text(tl, f'{place}   ·   rover {rv}  vs  alien {al}', (0.06, -0.15), 0.036, MUTED)
        self.t_turn = ui.text(tl, '', (0.06, -0.21), 0.036, TEXT)
        # top-centre: compass tape
        tc = root(b.a2dTopCenter, 'tc')
        self.compass = tc.attachNewNode('compass')
        self.compass.setZ(-0.07)
        ui.panel(self.compass, -0.7, 0.7, -0.045, 0.045, color=(0.04, 0.04, 0.1, 0.55), edge=None)
        self.compass_lines = self.compass.attachNewNode('lines')
        self.compass_labels = [ui.text(self.compass, '', (0, -0.012), 0.032, TEXT, TextNode.ACenter, shadow=False)
                               for _ in range(8)]
        self.compass_marks = [ui.text(self.compass, '', (0, -0.1), 0.04, CYAN, TextNode.ACenter, bold=True)
                              for _ in range(8)]
        self.t_heading = ui.text(self.compass, '', (0.78, -0.012), 0.034, CYAN, TextNode.ACenter, bold=True)
        # top-right: orbital map
        tr = root(b.a2dTopRight, 'tr')
        self.map_root = tr.attachNewNode('map')
        self.map_tex = DynamicTexture(220, 220, 'minimap')
        img = OnscreenImage(image=self.map_tex.tex, parent=self.map_root, pos=(-0.27, 0, -0.27), scale=0.23)
        img.setTransparency(TransparencyAttrib.MAlpha)
        self.map_root.attachNewNode(outline(-0.5, -0.04, -0.5, -0.04, CYAN, 1.5))
        self.t_cam = ui.text(self.map_root, '', (-0.27, -0.56), 0.032, CYAN, TextNode.ACenter)
        # bottom-centre: dashboard
        bc = root(b.a2dBottomCenter, 'bc')
        ui.panel(bc, -0.84, 0.84, 0.02, 0.37, color=(0.04, 0.04, 0.11, 0.86))
        ui.text(bc, 'FUEL', (-0.8, 0.31), 0.034, CYAN, bold=True)
        self.t_fuel = ui.text(bc, '', (-0.8, 0.19), 0.1, TEXT, bold=True)
        self.t_fuel0 = ui.text(bc, '', (-0.56, 0.19), 0.045, MUTED)
        self.segs = []
        for i in range(18):
            x0 = -0.8 + i * 0.031
            self.segs.append(DirectFrame(parent=bc, frameColor=GREEN, frameSize=(x0, x0 + 0.024, 0.11, 0.145)))
        self.t_route = ui.text(bc, '', (-0.8, 0.055), 0.032, TEXT)
        self.scope_tex = DynamicTexture(200, 200, 'scope')
        sc = OnscreenImage(image=self.scope_tex.tex, parent=bc, pos=(0, 0, 0.195), scale=0.165)
        sc.setTransparency(TransparencyAttrib.MAlpha)
        ui.text(bc, 'RADAR', (0, 0.025), 0.026, CYAN, TextNode.ACenter)
        ui.text(bc, 'ALIEN', (0.24, 0.31), 0.034, ORANGE, bold=True)
        self.pips = []
        for i in range(s.rules.alien_budget):
            x = 0.4 + i * 0.038
            self.pips.append(DirectFrame(parent=bc, frameColor=ORANGE, frameSize=(x - 0.013, x + 0.013, 0.31, 0.336)))
        self.t_alien = ui.text(bc, '', (0.24, 0.245), 0.032, TEXT)
        self.t_pulse = ui.text(bc, '', (0.24, 0.175), 0.032, TEXT)
        self.t_shield = ui.text(bc, '', (0.24, 0.11), 0.032, TEXT)
        self.t_auto = ui.text(bc, '', (0.24, 0.05), 0.032, MUTED)
        # bottom-left: log and AI thoughts
        bl = root(b.a2dBottomLeft, 'bl')
        self.t_ai = ui.text(bl, '', (0.05, 0.3), 0.032, PURPLE, wrap=24)
        self.t_log = [ui.text(bl, '', (0.05, 0.19 - i * 0.045), 0.03, MUTED) for i in range(4)]
        br = root(b.a2dBottomRight, 'br')
        self.t_help = ui.text(br, '', (-0.05, 0.05), 0.03, MUTED, TextNode.ARight)
        # left-centre: radio
        lc = root(b.a2dLeftCenter, 'lc')
        self.radio_root = lc.attachNewNode('radio')
        ui.panel(self.radio_root, 0.04, 1.02, -0.02, 0.22, color=(0.03, 0.06, 0.1, 0.82), edge=CYAN)
        self.t_speaker = ui.text(self.radio_root, '', (0.08, 0.15), 0.034, CYAN, bold=True)
        self.t_radio = ui.text(self.radio_root, '', (0.08, 0.09), 0.036, TEXT, wrap=25)
        self.radio_root.hide()
        self.queue = []
        self.current = None
        self.said = set()
        # centre: reticle, alerts
        a2d = root(b.aspect2d, 'centre')
        self.reticle = a2d.attachNewNode('reticle')
        ls = LineSegs()
        ls.setColor(*CYAN[:3], 0.6)
        ls.setThickness(1.5)
        for sx, sy in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
            ls.moveTo(sx * 0.4, 0, sy * 0.26)
            ls.drawTo(sx * 0.33, 0, sy * 0.26)
            ls.moveTo(sx * 0.4, 0, sy * 0.26)
            ls.drawTo(sx * 0.4, 0, sy * 0.19)
        ls.moveTo(-0.025, 0, 0.05)
        ls.drawTo(0.025, 0, 0.05)
        ls.moveTo(0, 0, 0.025)
        ls.drawTo(0, 0, 0.075)
        self.reticle.attachNewNode(ls.create())
        self.t_alert = ui.text(a2d, '', (0, 0.55), 0.075, RED, TextNode.ACenter, bold=True)
        self.t_banner = ui.text(a2d, '', (0, 0.45), 0.045, (1, 0.85, 0.75, 1), TextNode.ACenter)
        self.t_flash = ui.text(a2d, '', (0, -0.3), 0.045, (1, 0.75, 0.75, 1), TextNode.ACenter)
        self.flash_t = -10
        self.red = DirectFrame(parent=b.render2d, frameColor=(1, 0, 0, 0), frameSize=(-1, 1, -1, 1))
        self.roots.append(self.red)
        self.map_dirty = True

    def destroy(self):
        for n in self.roots:
            n.removeNode()
        self.roots = []

    def hide(self):
        for n in self.roots:
            n.hide()

    def show(self):
        for n in self.roots:
            n.show()

    # ------------------------------------------------------------ messages
    def flash(self, msg):
        self.t_flash.setText(msg)
        self.flash_t = self.g.now()

    def hurt(self):
        self.red['frameColor'] = (1, 0, 0, 0.38)

    def radio(self, key, once=True, **fmt):
        if once and key in self.said:
            return
        self.said.add(key)
        who, text = story.RADIO[key]
        self.queue.append((who, text.format(**fmt)))

    def update_radio(self, now):
        if self.current is None and self.queue:
            self.current = self.queue.pop(0) + (now,)
            self.g.sound.play('radio')
        if self.current is None:
            self.radio_root.hide()
            return
        who, text, t0 = self.current
        shown = int((now - t0) * 55)
        self.radio_root.show()
        self.t_speaker.setText(who)
        self.t_speaker['fg'] = SPEAKER.get(who, TEXT)
        self.t_radio.setText(text[:shown])
        hold = 2.6 if self.queue else 5.0
        if now - t0 > len(text) / 55 + hold:
            self.current = None

    # ------------------------------------------------------------ per frame
    def update(self, now):
        g, s = self.g, self.g.state
        rover_view = g.mode == 'rover'
        self.t_turn.setText(f'Turn {s.tick}' + ('   ·   PAUSED' if g.paused else ''))
        # fuel
        frac = max(0, s.fuel) / s.fuel0
        col = GREEN if frac > 0.45 else YELLOW if frac > 0.2 else RED
        self.t_fuel.setText(f'{max(s.fuel, 0)}')
        self.t_fuel['fg'] = col
        self.t_fuel0.setText(f'/ {s.fuel0}')
        lit = int(round(frac * len(self.segs)))
        for i, seg in enumerate(self.segs):
            seg['frameColor'] = col if i < lit else (0.16, 0.15, 0.26, 1)
        cost = g.costs[0] if rover_view else g.costs[1]
        label = 'Best known route' if rover_view else 'Cheapest route'
        self.t_route.setText(f'{label}: ' + ('none' if cost == INF else f'{cost:.0f}') +
                             f'   ·   shots {s.shots}   hits {s.hits}')
        self.t_route['fg'] = TEXT if cost != INF and cost <= s.fuel else RED
        # alien + systems
        for i, p in enumerate(self.pips):
            p['frameColor'] = ORANGE if i < s.budget else (0.3, 0.26, 0.3, 1)
        if s.result:
            msg = ''
        elif s.turn == ALIEN:
            msg = 'YOUR STRIKE: click the ground' if g.mode == 'alien' else 'Saucer is striking...'
        elif s.budget == 0:
            msg = 'Out of strikes'
        else:
            k = s.rules.alien_every - s.tick % s.rules.alien_every
            msg = f'Next strike in {k} turn{"s" if k > 1 else ""}'
        if s.pending:
            msg += f'  ·  {len(s.pending)} incoming'
        self.t_alien.setText(msg)
        if s.rules.scan_cost:
            ok = s.fuel > s.rules.scan_cost and s.turn != ALIEN and not s.result
            self.t_pulse.setText(f'[Q] RADAR PULSE   {s.rules.scan_cost} fuel' + ('' if ok else '  (wait)'))
            self.t_pulse['fg'] = CYAN if ok else DIM
        else:
            self.t_pulse.setText('RADAR PULSE   offline')
            self.t_pulse['fg'] = DIM
        if s.rules.shields:
            self.t_shield.setText('SHIELD   ' + ' '.join(['ON'] * s.shields + ['--'] * (s.rules.shields - s.shields)))
            self.t_shield['fg'] = CYAN if s.shields else DIM
        else:
            self.t_shield.setText('SHIELD   offline')
            self.t_shield['fg'] = DIM
        if g.waypoint:
            self.t_auto.setText(f'AUTOPILOT to {g.waypoint}   (any key stops)')
            self.t_auto['fg'] = GREEN
        else:
            self.t_auto.setText('Click the ground: autopilot' if rover_view else '')
            self.t_auto['fg'] = MUTED
        # thoughts, log, help
        lines = []
        if g.rover_ai:
            lines.append(f'{g.rover_ai.name}: ' + (g.rover_ai.info or f'route {max(0, len(g.plan) - 1)} tiles'))
        if g.alien_ai and not rover_view:
            lines.append(f'{g.alien_ai.name}: ' + (g.alien_ai.info or 'waiting'))
        self.t_ai.setText('\n'.join(lines))
        events = s.log if not rover_view else [(t, m) for t, m in s.log if 'drops a boulder' not in m]
        ev = list(reversed(events[-4:]))
        for i, t in enumerate(self.t_log):
            t.setText(f'T{ev[i][0]:>3}   {ev[i][1]}' if i < len(ev) else '')
            t['fg'] = TEXT if i == 0 else MUTED
        if g.mode == 'rover':
            h = ('W/S drive · A/D turn · Space wait · Q pulse' if g.cam != 'Overhead'
                 else 'Arrows drive N/S/W/E · Space wait · Q pulse')
        elif g.mode == 'alien':
            h = 'Left-click boulder · Right-click debris · Space hold'
        else:
            h = f'Space pause · → step · +/- speed ({g.speed}/s)'
        self.t_help.setText(h + '\nV camera · right-drag look · wheel zoom · Tab map · H help · Esc pause')
        self.t_cam.setText(f'{g.cam.upper()} CAM')
        self.reticle.show() if g.cam == 'Pilot' and not s.result else self.reticle.hide()
        # alerts
        route = set(g.plan[1:8])
        if s.result:
            self.t_alert.setText('')
            self.t_banner.setText('')
        elif s.rover in s.pending and s.pending[s.rover] == 1:
            self.t_alert.setText('INCOMING  -  MOVE!' if int(now * 4) % 2 else '')
            self.t_banner.setText('Debris lands on your position next turn')
        elif any(p in route for p in s.pending):
            self.t_alert.setText('')
            self.t_banner.setText('Debris incoming on the planned route')
        else:
            self.t_alert.setText('')
            self.t_banner.setText('')
        if now - self.flash_t > 2.0:
            self.t_flash.setText('')
        a = self.red['frameColor'][3]
        if a > 0:
            self.red['frameColor'] = (1, 0, 0, max(0, a - 0.015))
        self.update_compass()
        self.update_scope(now)
        if self.map_dirty:
            self.update_map()
            self.map_dirty = False
        self.update_radio(now)

    # ------------------------------------------------------------ compass
    def update_compass(self):
        g = self.g
        camh = g.base.camera.getH(g.base.render)
        hw = 0.66
        span = 80.0

        def x_of(h):
            rel = (h - camh + 180) % 360 - 180
            return None if abs(rel) > span else -rel / span * hw
        self.compass_lines.getChildren().detach()
        ls = LineSegs()
        ls.setThickness(1.2)
        ls.setColor(*MUTED[:3], 0.8)
        for a in range(0, 360, 10):
            x = x_of(a)
            if x is not None:
                big = a % 45 == 0
                ls.moveTo(x, 0, -0.045)
                ls.drawTo(x, 0, -0.045 + (0.03 if big else 0.015))
        ls.setColor(*CYAN)
        ls.moveTo(0, 0, -0.05)
        ls.drawTo(0, 0, 0.05)
        self.compass_lines.attachNewNode(ls.create())
        names = ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE']
        for i, t in enumerate(self.compass_labels):
            x = x_of(i * 45)
            if x is None or abs(x) < 0.03:
                t.setText('')
            else:
                t.setText(names[i])
                t.setX(x)
        marks = []
        s = g.state
        rx, ry = g.rover_np.getX(), g.rover_np.getY()

        def bearing(x, y):
            return math.degrees(math.atan2(-(x - rx), y - ry))
        px, py = g.world_xy(s.map.goal)
        marks.append(('H', CYAN, bearing(px, py)))
        for p in s.pending:
            x, y = g.world_xy(p)
            marks.append(('v', RED, bearing(x, y)))
        marks.append(('UFO', ORANGE, bearing(g.ufo_pos.x, g.ufo_pos.y)))
        for i, t in enumerate(self.compass_marks):
            if i < len(marks):
                label, col, h = marks[i]
                x = x_of(h)
                if x is None:
                    x = hw if ((h - camh + 180) % 360 - 180) < 0 else -hw
                    label = ('>' if x > 0 else '<') + label if label == 'H' else ''
                t.setText(label)
                t['fg'] = col
                t.setX(x)
                t.setZ(-0.15 if label == 'UFO' else -0.1)
            else:
                t.setText('')
        names4 = {0: 'N', 90: 'W', 180: 'S', 270: 'E'}
        hdg = int(round((-camh) % 360))
        self.t_heading.setText(f'{hdg:03d}°')

    # ------------------------------------------------------------ radar scope
    def update_scope(self, now):
        import pygame as pg
        g, s = self.g, self.g.state
        S = 200
        c = S / 2
        R = S / 2 - 4
        surf = pg.Surface((S, S), pg.SRCALPHA)
        pg.draw.circle(surf, (6, 26, 32, 235), (c, c), R)
        for f in (1 / 3, 2 / 3):
            pg.draw.circle(surf, (40, 120, 130, 160), (c, c), R * f, 1)
        pg.draw.circle(surf, (86, 220, 232, 255), (c, c), R, 2)
        radar = s.rules.radar + 0.5
        k = R / radar
        camh = math.radians(g.base.camera.getH(g.base.render))
        ch, sh = math.cos(-camh), math.sin(-camh)
        rx, ry = g.rover_np.getX(), g.rover_np.getY()

        def to_scope(x, y):
            dx, dy = x - rx, y - ry
            lx, ly = dx * ch - dy * sh, dx * sh + dy * ch
            return c + lx * k, c - ly * k, math.hypot(lx, ly)
        r0, c0 = s.rover
        n = s.map.size
        for r in range(max(0, r0 - 6), min(n, r0 + 7)):
            for cc in range(max(0, c0 - 6), min(n, c0 + 7)):
                if s.map.walls[r][cc]:
                    x, y, d = to_scope(cc, -r)
                    if d < radar - 0.3:
                        pg.draw.circle(surf, (70, 96, 100, 200), (x, y), k * 0.42)
        blk = s.belief if g.mode == 'rover' else s.blk
        for p, b in blk.items():
            x, y, d = to_scope(p[1], -p[0])
            if d < radar - 0.3:
                pg.draw.circle(surf, (240, 150, 80) if b == SOFT else (255, 80, 40), (x, y), k * 0.36)
        pulse = 0.5 + 0.5 * math.sin(now * 8)
        for p in s.pending:
            x, y, d = to_scope(p[1], -p[0])
            if d < radar:
                pg.draw.circle(surf, (255, 80, 70), (x, y), k * (0.4 + 0.2 * pulse), 2)
        gx, gy = g.world_xy(s.map.goal)
        x, y, d = to_scope(gx, gy)
        if d < radar:
            pg.draw.circle(surf, (86, 220, 232), (x, y), k * 0.5, 2)
        else:
            ang = math.atan2(y - c, x - c)
            ex, ey = c + (R - 9) * math.cos(ang), c + (R - 9) * math.sin(ang)
            pg.draw.circle(surf, (86, 220, 232), (ex, ey), 6, 2)
        a = now * 2.2
        pg.draw.line(surf, (86, 220, 232, 140), (c, c), (c + R * math.cos(a), c + R * math.sin(a)), 2)
        pg.draw.polygon(surf, (255, 255, 255), [(c, c - 9), (c - 6, c + 6), (c + 6, c + 6)])
        self.scope_tex.load(surf)

    # ------------------------------------------------------------ orbital map
    def update_map(self):
        import pygame as pg
        g, s = self.g, self.g.state
        n = s.map.size
        size = 220
        if getattr(g, '_map_seed', None) != g.seed:
            img = g.fields['image']
            surf = pg.surfarray.make_surface(img.transpose(1, 0, 2))
            g._map_base = pg.transform.smoothscale(surf, (size, size))
            g._map_seed = g.seed
        m = pg.Surface((size, size), pg.SRCALPHA)
        m.blit(g._map_base, (0, 0))
        k = size / n

        def c(p):
            return ((p[1] + 0.5) * k, (p[0] + 0.5) * k)
        if len(g.plan) > 1:
            pg.draw.lines(m, (90, 225, 235), False, [c(p) for p in g.plan], 2)
        blk = s.belief if g.mode == 'rover' else s.blk
        for p, b in blk.items():
            pg.draw.circle(m, (240, 150, 80) if b == SOFT else (255, 80, 40), c(p), k * 0.4)
        for p in s.pending:
            pg.draw.circle(m, (255, 80, 70), c(p), k * 0.55, 2)
        if g.waypoint:
            pg.draw.circle(m, (120, 255, 150), c(g.waypoint), k * 0.5, 2)
        pg.draw.circle(m, (90, 225, 235), c(s.map.goal), k * 0.75, 2)
        pg.draw.circle(m, (90, 225, 235), c(s.rover), (s.rules.radar + 0.5) * k, 1)
        x, y = c(s.rover)
        pg.draw.circle(m, (255, 255, 255), (x, y), k * 0.45)
        self.map_tex.load(m)
