"""Sound effects and ambient music, synthesised with numpy on first run and
cached as WAV files in ui3d/sfx/. No downloaded assets."""
import os
import wave

import numpy as np

RATE = 22050
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sfx')


def _t(sec):
    return np.arange(int(RATE * sec)) / RATE


def _env(n, attack=0.01, release=0.2):
    e = np.ones(n)
    a, r = int(RATE * attack), int(RATE * release)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def _noise(n, rng, smooth=1):
    x = rng.normal(size=n)
    if smooth > 1:
        x = np.convolve(x, np.ones(smooth) / smooth, mode='same')
    return x / (np.abs(x).max() + 1e-9)


def _chirp(f0, f1, sec):
    t = _t(sec)
    f = np.geomspace(f0, f1, len(t))
    return np.sin(2 * np.pi * np.cumsum(f) / RATE)


def make_all():
    rng = np.random.default_rng(5)
    s = {}
    t = _t(0.28)
    s['laser'] = (0.6 * _chirp(1800, 260, 0.28) + 0.25 * np.sign(_chirp(900, 130, 0.28))) * np.exp(-t * 9)
    t = _t(1.1)
    s['boom'] = (_noise(len(t), rng, 40) * 0.9 + 0.5 * np.sin(2 * np.pi * 48 * t)) * np.exp(-t * 4.5)
    t = _t(0.5)
    s['thud'] = (_noise(len(t), rng, 60) + 0.6 * np.sin(2 * np.pi * 70 * t)) * np.exp(-t * 10)
    beep = np.sin(2 * np.pi * 980 * _t(0.09)) * _env(int(RATE * 0.09), 0.005, 0.03)
    s['alarm'] = np.concatenate([beep, np.zeros(int(RATE * 0.07)), beep, np.zeros(int(RATE * 0.07)), beep])
    t = _t(0.18)
    s['click'] = np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 60)
    t = _t(0.35)
    s['radio'] = (_noise(len(t), rng, 3) * 0.35 + 0.3 * np.sin(2 * np.pi * 1200 * t)) * _env(len(t), 0.01, 0.1)
    t = _t(1.0)
    hum = 0.5 * np.sin(2 * np.pi * 82 * t) + 0.25 * np.sin(2 * np.pi * 164 * t) + 0.2 * _noise(len(t), rng, 25)
    s['motor'] = hum * 0.6
    t = _t(0.8)
    s['scan'] = 0.6 * np.sin(2 * np.pi * (600 + 900 * t) * t) * np.exp(-t * 3)
    t = _t(0.6)
    s['whoosh'] = _noise(len(t), rng, 8) * np.sin(np.pi * t / 0.6) * 0.8
    t = _t(0.5)
    s['shield'] = 0.6 * np.sin(2 * np.pi * 440 * t) * np.sin(2 * np.pi * 7 * t) * np.exp(-t * 4)
    t = _t(5.0)
    s['launch'] = (_noise(len(t), rng, 30) * np.minimum(1, t / 0.8) * np.exp(-np.maximum(0, t - 3.5) * 2)
                   + 0.4 * np.sin(2 * np.pi * (40 + 25 * t) * t) * np.minimum(1, t))
    notes = [523, 659, 784, 1046]
    s['win'] = np.concatenate([np.sin(2 * np.pi * f * _t(0.16)) * _env(int(RATE * 0.16), 0.01, 0.08) for f in notes])
    s['lose'] = np.concatenate([np.sin(2 * np.pi * f * _t(0.3)) * _env(int(RATE * 0.3), 0.01, 0.2)
                                for f in (392, 330, 262)])
    # ambient drone: slow detuned pad, loops cleanly over 16 s
    t = _t(16.0)
    drone = np.zeros_like(t)
    for f, a in ((55, 0.5), (55.4, 0.4), (82.5, 0.3), (110.2, 0.18), (164.8, 0.1)):
        drone += a * np.sin(2 * np.pi * f * t)
    drone *= 0.6 + 0.4 * np.sin(2 * np.pi * t / 16)
    drone += 0.05 * _noise(len(t), rng, 200)
    s['ambient'] = drone * 0.5
    return s


def _write(path, x):
    x = np.clip(x / (np.abs(x).max() + 1e-9) * 0.85, -1, 1)
    data = (x * 32767).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data)


def ensure_files():
    names = ['laser', 'boom', 'thud', 'alarm', 'click', 'radio', 'motor', 'scan', 'whoosh', 'shield',
             'launch', 'win', 'lose', 'ambient']
    if all(os.path.exists(os.path.join(DIR, n + '.wav')) for n in names):
        return {n: os.path.join(DIR, n + '.wav') for n in names}
    os.makedirs(DIR, exist_ok=True)
    out = {}
    for n, x in make_all().items():
        p = os.path.join(DIR, n + '.wav')
        _write(p, x)
        out[n] = p
    return out


class Sound:
    """Thin wrapper: silently does nothing if the audio device is unavailable."""

    VOLUME = {'motor': 0.25, 'ambient': 0.35, 'radio': 0.35, 'click': 0.4, 'alarm': 0.5}

    def __init__(self, base):
        from panda3d.core import Filename
        self.base = base
        self.muted = False
        self.sounds = {}
        try:
            for name, path in ensure_files().items():
                snd = base.loader.loadSfx(Filename.fromOsSpecific(path).getFullpath())
                if snd:
                    snd.setVolume(self.VOLUME.get(name, 0.7))
                    self.sounds[name] = snd
        except Exception:
            self.sounds = {}
        if 'ambient' in self.sounds:
            self.sounds['ambient'].setLoop(True)
            self.sounds['ambient'].play()
        if 'motor' in self.sounds:
            self.sounds['motor'].setLoop(True)

    def play(self, name):
        s = self.sounds.get(name)
        if s and not self.muted:
            s.play()

    def motor(self, on):
        s = self.sounds.get('motor')
        if not s:
            return
        playing = s.status() == s.PLAYING
        if on and not playing and not self.muted:
            s.play()
        elif (not on or self.muted) and playing:
            s.stop()

    def toggle_mute(self):
        self.muted = not self.muted
        amb = self.sounds.get('ambient')
        if amb:
            amb.stop() if self.muted else amb.play()
        self.motor(False)
        return self.muted
