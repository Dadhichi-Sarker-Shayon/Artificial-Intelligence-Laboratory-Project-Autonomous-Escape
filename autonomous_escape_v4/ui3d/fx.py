"""Version 4 effects: laser shots that shatter obstacles, debris that is fired
into the sky, falls back as a burning meteor and blasts a crater, tractor-beam
boulder drops, dust, and the saucer's flight. Mixed into the Game class."""
import math
import random

import numpy as np
from panda3d.core import CardMaker, LineSegs, TextNode, Vec3

from escape import SOFT
from . import models

GRAVITY = 2.0                                           # low-gravity arcs (Phobos)
METEOR_DIR = Vec3(0.45, 0.55, 0.72).normalized()       # debris falls in from the north-east sky


class Effects:
    # ------------------------------------------------------------ plumbing
    def add_effect(self, node, dur, fn, delay=0.0):
        if delay:
            node.hide()
        self.effects.append([node, self.now() + delay, dur, fn])

    def update_effects(self, now):
        keep = []
        for e in self.effects:
            node, t0, dur, fn = e
            if now < t0:
                keep.append(e)
                continue
            if node.isHidden() and not node.hasPythonTag('keep_hidden'):
                node.show()
            t = (now - t0) / dur
            if t >= 1:
                node.removeNode()
                continue
            fn(node, t)
            keep.append(e)
        self.effects = keep
        hidden = set()
        for e in self.effects:
            if not e[0].isEmpty() and e[0].hasPythonTag('tile'):
                hidden.add(e[0].getPythonTag('tile'))
        for p, (node, _) in self.boulder_nodes.items():
            node.hide() if p in hidden else node.show()

    def glowcard(self, parent, size, color, billboard=True):
        return models.glow_card(parent, self.glow, size, color, billboard)

    def puff(self, parent, size, color):
        """Soft dust / smoke sprite: alpha-blended, not glowing."""
        cm = CardMaker('puff')
        cm.setFrame(-size / 2, size / 2, -size / 2, size / 2)
        c = parent.attachNewNode(cm.generate())
        c.setTexture(self.glow)
        c.setColor(*color)
        c.setBillboardPointEye()
        self.meshes.translucent(c)
        c.setLightOff(1)
        c.setShaderOff(1)
        return c

    def ground_pos(self, p, lift=0.0):
        x, y = self.world_xy(p)
        return Vec3(x, y, self.gz(x, y) + lift)

    def shake_from(self, pos, strength):
        d = (self.rover_np.getPos() - pos).length()
        self.shake = max(self.shake, strength * max(0.0, 1 - d / 14))

    # ------------------------------------------------------------ dust behind the wheels
    def emit_dust(self):
        h = math.radians(self.rover_np.getH())
        fx, fy = -math.sin(h), math.cos(h)
        root = self.dyn.attachNewNode('dust')
        puffs = []
        for side in (-1, 1):
            x = self.rover_np.getX() - fx * 0.24 + fy * side * 0.25
            y = self.rover_np.getY() - fy * 0.24 - fx * side * 0.25
            c = self.puff(root, 0.22, (0.55, 0.5, 0.45, 0.35))
            c.setPos(x, y, self.gz(x, y) + 0.06)
            puffs.append((c, Vec3(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(0.05, 0.15))))

        def fn(node, t):
            for c, v in puffs:
                c.setPos(c.getPos() + v * 0.016)
                c.setScale(1 + 2.5 * t)
            node.setAlphaScale(1 - t)
        self.add_effect(root, 1.3, fn)

    # ------------------------------------------------------------ laser
    def fire_laser(self, p, dying, style):
        """Charge at the mast, beam, the obstacle heats up and shatters into
        fragments that drift away in the weak gravity."""
        root = self.dyn.attachNewNode('laser')
        tgt = self.ground_pos(p, 0.22)
        head = self.rover.head
        charge = self.glowcard(head, 0.05, (1, 0.35, 0.45, 1))
        heat = self.glowcard(root, 0.4, (1, 0.5, 0.2, 1))
        heat.setPos(tgt)
        heat.hide()
        beam_np = root.attachNewNode('beam')
        state = {'shattered': False, 'frags': [], 'puffs': []}
        rng = np.random.default_rng(p[0] * 31 + p[1])
        dur = 2.2

        def fn(node, t):
            T = t * dur
            if not charge.isEmpty():
                charge.setScale(1 + min(T, 0.25) * 30 if T < 0.6 else 0.01)
            beam_np.getChildren().detach()
            if 0.25 <= T < 0.62:
                src = head.getPos(self.base.render) + Vec3(0, 0, 0.0)
                wob = 1 + 0.3 * math.sin(T * 90)
                for th, col in ((9 * wob, (1, 0.25, 0.4, 0.5)), (4, (1, 0.45, 0.55, 1)), (1.6, (1, 0.95, 0.95, 1))):
                    ls = LineSegs()
                    ls.setThickness(th)
                    ls.setColor(*col)
                    ls.moveTo(src)
                    ls.drawTo(tgt)
                    self.meshes.additive(beam_np.attachNewNode(ls.create()))
                heat.show()
                heat.setScale(1 + (T - 0.25) * 6)
            if T >= 0.5 and not state['shattered']:
                state['shattered'] = True
                if dying is not None and not dying.isEmpty():
                    dying.removeNode()
                self.sound.play('shatter')
                for k in range(12):
                    frag = models.fragment_model(style, rng)
                    frag.reparentTo(root)
                    frag.setPos(tgt)
                    a = rng.uniform(0, 2 * math.pi)
                    v = Vec3(math.cos(a) * rng.uniform(0.4, 1.3), math.sin(a) * rng.uniform(0.4, 1.3),
                             rng.uniform(0.8, 2.0))
                    spin = Vec3(*rng.uniform(-400, 400, 3))
                    state['frags'].append([frag, v, spin])
                for k in range(8):
                    c = self.puff(root, 0.35, (0.6, 0.52, 0.45, 0.45))
                    c.setPos(tgt + Vec3(rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0))
                    state['puffs'].append((c, Vec3(*rng.uniform(-0.3, 0.3, 2), rng.uniform(0.1, 0.4))))
                self.add_burst(tgt, (1, 0.7, 0.4, 1), 0.7)
            if T >= 0.62:
                heat.setScale(max(0.01, heat.getScale().x - 0.2))
            dt = 1 / 60
            for f in state['frags']:
                frag, v, spin = f
                pos = frag.getPos() + v * dt
                v.z -= GRAVITY * dt
                gz = self.gz(pos.x, pos.y) + 0.02
                if pos.z < gz:
                    pos.z = gz
                    v *= 0.3
                    v.z = abs(v.z) * 0.3
                frag.setPos(pos)
                frag.setHpr(frag.getHpr() + spin * dt)
                if T > 1.5:
                    frag.setScale(frag.getScale() * 0.96)
            for c, v in state['puffs']:
                c.setPos(c.getPos() + v * dt)
                c.setScale(c.getScale() * 1.012)
                c.setAlphaScale(max(0, 1 - T / dur))
        root.setPythonTag('tile', p)
        self.add_effect(root, dur, fn)
        self.add_effect(charge, 0.62, lambda n, t: None)

    def add_burst(self, pos, color, dur):
        root = self.dyn.attachNewNode('burst')
        bits = []
        for _ in range(14):
            c = self.glowcard(root, 0.08, color)
            bits.append((c, Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.3, 1.6))))

        def fn(node, t):
            for c, v in bits:
                c.setPos(pos + v * t * 1.1 - Vec3(0, 0, 0.6 * t * t))
            node.setAlphaScale(1 - t)
        self.add_effect(root, dur, fn)

    # ------------------------------------------------------------ debris: launch, flight, plunge, impact
    def launch_debris(self, style):
        root = self.dyn.attachNewNode('launch')
        proj, body = models.projectile_model(style, self.glow)
        proj.reparentTo(root)
        proj.hide()
        start = Vec3(self.ufo_pos)
        sau = self.saucer
        trail = []
        self.sound.play('charge')
        fired = {'done': False}

        def fn(node, t):
            T = t * 1.3
            if T < 0.4:
                sau.charge.setScale(0.01 + T * 3)
                return
            sau.charge.setScale(max(0.01, 1.2 - (T - 0.4) * 4))
            if not fired['done']:
                fired['done'] = True
                proj.show()
                self.sound.play('whoosh')
            u = (T - 0.4) / 0.9
            pos = start + Vec3(1.5 * u, 2.2 * u, -0.2 + 16 * u * u)
            proj.setPos(pos)
            body.setHpr(T * 400, T * 250, 0)
            if len(trail) < 18:
                c = self.puff(root, 0.25, (0.55, 0.5, 0.48, 0.4))
                c.setPos(pos)
                trail.append(c)
            for c in trail:
                c.setScale(c.getScale() * 1.03)
            root.setAlphaScale(max(0, 1 - max(0, u - 0.7) * 3))
        self.add_effect(root, 1.3, fn)

    def update_meteors(self, now):
        """Incoming debris high in the sky, sinking a little each turn, with a
        red landing ring and a countdown on the ground."""
        s = self.state
        for p in list(self.meteors):
            if p not in s.pending:
                for n in self.meteors[p][:2]:
                    n.removeNode()
                del self.meteors[p]
        prog = min(1.0, (now - self.last_tick) / 0.8) * 0.6
        for p, ttl in s.pending.items():
            x, y = self.world_xy(p)
            gz = self.gz(x, y)
            if p not in self.meteors:
                style = self.proj_style.get(p, 'rock')
                head, body = models.projectile_model(style, self.glow)
                head.reparentTo(self.dyn)
                heat = head.getPythonTag('heat')
                for i in range(1, 12):
                    c = self.glowcard(head, 0.6 - i * 0.04, heat[:3] + (0.5 - i * 0.04,))
                    c.setPos(METEOR_DIR * i * 0.25)
                ring = self.meshes.additive(self.meshes.ring('target', 0.36, 0.44, (0.95, 0.3, 0.3, 1)))
                ring.reparentTo(self.dyn)
                ring.setPos(x, y, gz + 0.05)
                self.meshes.additive(self.meshes.ring('target', 0.08, 0.14, (0.95, 0.3, 0.3, 1))).reparentTo(ring)
                label = TextNode('count')
                label.setFont(self.ui.bold)
                label.setAlign(TextNode.ACenter)
                label.setTextColor(1, 0.45, 0.4, 1)
                tn = ring.attachNewNode(label)
                tn.setScale(0.35)
                tn.setPos(0, 0, 0.6)
                tn.setBillboardPointEye()
                tn.setLightOff(1)
                tn.setShaderOff(1)
                tn.setDepthTest(False)
                tn.setBin('fixed', 20)
                self.meteors[p] = [head, ring, body, label]
            head, ring, body, label = self.meteors[p]
            head.setPos(Vec3(x, y, gz) + METEOR_DIR * (3 + 7 * (ttl - prog)))
            body.setHpr(now * 120, now * 80, 0)
            ring.setScale(1 + 0.15 * math.sin(now * 8))
            label.setText(str(ttl))

    def plunge(self, p, style):
        """The final fall: from high in the sky to the ground in half a second."""
        x, y = self.world_xy(p)
        gz = self.gz(x, y)
        old = self.meteors.get(p)
        start = old[0].getPos() if old else Vec3(x, y, gz) + METEOR_DIR * 10
        root = self.dyn.attachNewNode('plunge')
        root.setPythonTag('tile', p)
        proj, body = models.projectile_model(style, self.glow)
        proj.reparentTo(root)
        heat = proj.getPythonTag('heat')
        end = Vec3(x, y, gz + 0.1)
        smoke = []

        def fn(node, t):
            pos = start + (end - start) * (t * t)
            proj.setPos(pos)
            body.setHpr(t * 700, t * 500, 0)
            c = self.puff(root, 0.32, (0.5, 0.46, 0.44, 0.5))
            c.setPos(pos)
            smoke.append(c)
            f = self.glowcard(root, 0.4, heat[:3] + (0.6,))
            f.setPos(pos)
            smoke.append(f)
            for k, c in enumerate(smoke):
                c.setScale(c.getScale() * 1.02)
        self.add_effect(root, 0.5, fn)
        self.impact(p, style, delay=0.5)

    def impact(self, p, style, delay=0.0):
        pos = self.ground_pos(p, 0.12)
        root = self.dyn.attachNewNode('impact')
        root.setPythonTag('tile', p)
        flash = self.glowcard(root, 3.2, (1, 0.75, 0.4, 1))
        flash.setPos(pos)
        ring = self.meshes.translucent(self.meshes.ring('shock', 0.85, 1.0, (0.8, 0.72, 0.62, 0.7)))
        ring.setLightOff(1)
        ring.reparentTo(root)
        ring.setPos(pos)
        rng = np.random.default_rng(p[0] * 97 + p[1] + int(self.now() * 10))
        bits = []
        for k in range(10):
            frag = models.fragment_model(style if rng.random() < 0.5 else 'rock', rng)
            frag.reparentTo(root)
            frag.setPos(pos)
            a = rng.uniform(0, 2 * math.pi)
            bits.append([frag, Vec3(math.cos(a) * rng.uniform(0.5, 1.6), math.sin(a) * rng.uniform(0.5, 1.6),
                                    rng.uniform(1.0, 2.6)), Vec3(*rng.uniform(-500, 500, 3))])
        plume = []
        for k in range(16):
            c = self.puff(root, 0.5, (0.58, 0.52, 0.47, 0.55))
            c.setPos(pos)
            a = rng.uniform(0, 2 * math.pi)
            plume.append((c, Vec3(math.cos(a) * rng.uniform(0.3, 1.0), math.sin(a) * rng.uniform(0.3, 1.0),
                                  rng.uniform(0.2, 0.9))))
        state = {'started': False}

        def fn(node, t):
            if not state['started']:
                state['started'] = True
                node.clearPythonTag('tile')
                self.sound.play('boom')
                self.shake_from(pos, 0.9)
                self.scorch(pos)
            T = t * 2.0
            flash.setScale(max(0.01, 1 - T * 2.5))
            ring.setScale(0.3 + 2.2 * min(1, T * 1.4))
            ring.setAlphaScale(max(0, 1 - T * 1.2))
            dt = 1 / 60
            for b in bits:
                frag, v, spin = b
                q = frag.getPos() + v * dt
                v.z -= GRAVITY * dt
                g = self.gz(q.x, q.y) + 0.02
                if q.z < g:
                    q.z, v.x, v.y, v.z = g, v.x * 0.4, v.y * 0.4, 0
                frag.setPos(q)
                frag.setHpr(frag.getHpr() + spin * dt)
            for c, v in plume:
                c.setPos(c.getPos() + v * dt)
                v *= 0.985
                c.setScale(1 + 3 * T)
                c.setAlphaScale(max(0, 1 - T / 2))
        self.add_effect(root, 2.0, fn, delay=delay)

    def scorch(self, pos):
        d = self.meshes.disc('scorch', 0.6, (0.05, 0.04, 0.04, 0.45))
        self.meshes.translucent(d)
        d.setLightOff(1)
        d.setShaderOff(1)
        d.setPos(pos.x, pos.y, pos.z - 0.08)
        d.reparentTo(self.decals)

    # ------------------------------------------------------------ boulder drop on the tractor beam
    def drop_obstacle(self, p, style):
        from escape import SOFT as _S
        tgt = self.ground_pos(p)
        root = self.dyn.attachNewNode('drop')
        root.setPythonTag('tile', p)
        rock = models.obstacle_model(style, False, (p[0] * 7 + p[1] * 13) % 8, self.glow)
        rock.reparentTo(root)
        rock.hide()
        beam = self.meshes.additive(self.meshes.cone('beam', 0.3, 0.75, 1.0, (0.7, 1, 0.8, 0.22)))
        beam.reparentTo(root)
        beam.hide()
        rings = []
        for k in range(3):
            r = self.meshes.additive(self.meshes.ring('beamring', 0.32, 0.4, (0.6, 1, 0.8, 0.6)))
            r.reparentTo(root)
            r.hide()
            rings.append(r)
        state = {'landed': False}
        dur = 1.8

        def fn(node, t):
            T = t * dur
            u = Vec3(self.ufo_pos)
            near = (Vec3(u.x, u.y, 0) - Vec3(tgt.x, tgt.y, 0)).length() < 1.2
            if T < 0.35 and not near:
                return
            k = min(1.0, max(0.0, (T - 0.35) / 0.9))
            h = max(0.3, u.z - tgt.z - 0.15)
            beam.show()
            beam.setPos(u.x, u.y, u.z - 0.15)
            beam.setScale(1, 1, h)
            beam.setAlphaScale(1 - max(0, T - 1.3) * 2)
            for i, r in enumerate(rings):
                r.show()
                f = ((T * 0.8 + i / 3) % 1)
                r.setPos(u.x + (tgt.x - u.x) * f, u.y + (tgt.y - u.y) * f, u.z - 0.15 - h * f)
                r.setScale(1 + f)
            rock.show()
            e = k * k * (3 - 2 * k)
            rock.setPos(u.x + (tgt.x - u.x) * e, u.y + (tgt.y - u.y) * e, (u.z - 0.4) + (tgt.z - (u.z - 0.4)) * e)
            rock.setH(T * 90)
            if k >= 1 and not state['landed']:
                state['landed'] = True
                node.clearPythonTag('tile')
                rock.hide()
                self.sound.play('thud')
                self.shake_from(tgt, 0.3)
                for j in range(8):
                    c = self.puff(node, 0.4, (0.58, 0.52, 0.47, 0.5))
                    a = j * 0.785
                    c.setPos(tgt + Vec3(math.cos(a) * 0.3, math.sin(a) * 0.3, 0.08))
        self.add_effect(root, dur, fn)

    # ------------------------------------------------------------ scan + shield
    def add_scan(self):
        s = self.state
        root = self.dyn.attachNewNode('scan')
        root.setPos(self.rover_np.getPos() + Vec3(0, 0, 0.08))
        ring = self.meshes.additive(self.meshes.ring('pulse', 0.9, 1.0, (0.34, 0.86, 0.91, 1)))
        ring.reparentTo(root)
        R = s.rules.scan_radius + 0.5

        def fn(node, t):
            ring.setScale(0.2 + R * (t * t * (3 - 2 * t)))
            node.setAlphaScale(1 - t)
        self.add_effect(root, 1.0, fn)

    def add_shield_flash(self):
        root = self.dyn.attachNewNode('shieldflash')
        root.setPos(self.rover_np.getPos() + Vec3(0, 0, 0.25))
        flash = self.glowcard(root, 1.6, (0.5, 0.95, 1, 1))

        def fn(node, t):
            flash.setScale(1 + t)
            node.setAlphaScale(1 - t)
        self.add_effect(root, 0.7, fn)

    # ------------------------------------------------------------ the saucer's flight
    def update_ufo(self, now, dt):
        """Steered flight: stalks the rover, rushes to strike points, banks into
        turns, bobs, spins its light ring; its shadow follows on the ground."""
        rx, ry = self.rover_np.getX(), self.rover_np.getY()
        striking = self.ufo_goal is not None and now < self.ufo_until
        if striking:
            goal = self.ufo_goal
            vmax = 11.0
        else:
            a = now * 0.22
            goal = Vec3(rx + 7 * math.cos(a), ry + 7 * math.sin(a), self.gz(rx, ry) + 4.6 + 0.6 * math.sin(now * 0.7))
            vmax = 4.5
        to = goal - self.ufo_pos
        dist = to.length()
        want = to.normalized() * min(vmax, dist * 2.2) if dist > 1e-4 else Vec3(0)
        acc = want - self.ufo_vel
        amax = 14.0 * dt
        if acc.length() > amax:
            acc = acc.normalized() * amax
        self.ufo_vel += acc
        self.ufo_pos += self.ufo_vel * dt
        bob = 0.07 * math.sin(now * 2.3)
        sau = self.saucer
        sau.root.setPos(self.ufo_pos + Vec3(0, 0, bob))
        v = self.ufo_vel
        sau.tilt.setHpr(0, max(-25, min(25, -v.y * 2.6 - acc.y * 20)), max(-25, min(25, v.x * 2.6 + acc.x * 20)))
        sau.animate(now, self.ufo_vel.length())
        sau.emitter.setAlphaScale(0.5 + 0.3 * math.sin(now * 5))
        gz = self.gz(self.ufo_pos.x, self.ufo_pos.y)
        self.ufo_shadow.setPos(self.ufo_pos.x, self.ufo_pos.y, gz + 0.04)
        alt = max(0.5, self.ufo_pos.z - gz)
        self.ufo_shadow.setScale(1.1 + alt * 0.08)
        self.ufo_shadow.setAlphaScale(max(0.15, 0.7 - alt * 0.07))
