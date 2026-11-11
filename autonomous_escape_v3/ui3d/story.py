"""Campaign: missions, briefings, radio chatter and the save file.

The campaign only changes rule *parameters* (alien AI, strikes, fuel and the
rover's optional abilities); the rules engine is the same one the AI
benchmark uses.
"""
import json
import os
from dataclasses import dataclass, field, replace

from escape.config import DEFAULT

SAVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'save.json')

PROLOGUE = (
    'Year 2189. Stickney Crater, Phobos.',
    'An asteroid has torn through the mining outpost. The crew got out on the last shuttle,',
    'all but one machine: PATHFINDER-7, the rover carrying the outpost\'s core samples.',
    'A relief shuttle can hold on a single launch pad for a short window. Between you and it',
    'lies a debris field, and something in orbit that does not want you to leave.',
)

EPILOGUE = (
    'Shuttle away. PATHFINDER-7 and the Stickney core samples are climbing out of Phobos.',
    'Behind them the saucer circles an empty pad.',
    'CONTROL: "Welcome home, Pathfinder. Drinks are on me."',
)


@dataclass(frozen=True)
class Mission:
    key: str
    title: str
    place: str
    seed: int
    alien: str
    briefing: tuple
    budget: int
    fuel_mult: float
    scan_cost: int = 0
    shields: int = 0
    stars: tuple = (35, 55)          # escape score needed for 2 and 3 stars
    unlock: str = ''                 # new system announced in this mission

    def rules(self):
        return replace(DEFAULT, alien_budget=self.budget, fuel_mult=self.fuel_mult,
                       scan_cost=self.scan_cost, shields=self.shields)


MISSIONS = [
    Mission('m1', 'First Light', 'Stickney Rim', 21, 'Random', (
        'CONTROL: Pathfinder, you are the last one out. The relief shuttle is waiting',
        'on pad H in the crater to the south-east.',
        'Up / W drives forward, Down / S reverses, Left / Right turn on the spot.',
        'Loose boulders glow warm: drive into one and the laser cuts through (2 fuel).',
        'Every move costs 1 fuel. And keep an eye on the sky: something is up there.',
    ), budget=4, fuel_mult=2.4, stars=(40, 60)),
    Mission('m2', 'The Grooves', 'Kepler Dorsum', 42, 'Greedy', (
        'CONTROL: That thing in orbit is a saucer, and it is dropping boulders on the',
        'routes ahead of you. Outside your radar you will not see them land.',
        'NEW SYSTEM: RADAR PULSE (Q). One pulse maps every boulder within 10 tiles',
        'for 2 fuel. Use it before you commit to a long corridor.',
    ), budget=6, fuel_mult=2.2, scan_cost=2, stars=(35, 55), unlock='Radar pulse (Q)'),
    Mission('m3', 'Meteor Rain', 'Limtoc Crater', 79, 'Core', (
        'CONTROL: New behaviour. The saucer is flinging debris. Watch for the red',
        'landing rings: debris lands two turns later, fuses boulders into laser-proof',
        'rock and hits the rover hard. This alien finds the narrowest choke points.',
        'NEW SYSTEM: DEBRIS SHIELD. It absorbs one hit, then it is gone.',
    ), budget=8, fuel_mult=2.0, scan_cost=2, shields=1, stars=(30, 50), unlock='Debris shield (1 hit)'),
    Mission('m4', 'The Warden', 'Reldresal', 104, 'MCTS', (
        'CONTROL: It is learning. The saucer now plays out thousands of futures before',
        'every strike. It will try to drain your fuel with walls you have to blast.',
        'Keep options open, pulse the radar early, and never wait under a red ring.',
    ), budget=8, fuel_mult=2.0, scan_cost=2, shields=1, stars=(30, 50)),
    Mission('m5', 'Last Launch', 'Stickney Floor', 77, 'Core', (
        'CONTROL: This is the last window, Pathfinder. The shuttle lifts with or without',
        'you, and the saucer is back to sealing choke points, with ten strikes left.',
        'The direct routes will close. Pulse early, blast fast, and do not get fenced in.',
        'Engineering found a second shield cell. Get those samples home.',
    ), budget=10, fuel_mult=2.0, scan_cost=2, shields=2, stars=(25, 40)),
]


def stars_for(mission, won, score):
    if not won:
        return 0
    lo, hi = mission.stars
    return 3 if score >= hi else 2 if score >= lo else 1


def load_progress():
    try:
        with open(SAVE, encoding='utf8') as f:
            data = json.load(f)
        return {k: int(v) for k, v in data.get('stars', {}).items()}
    except (OSError, ValueError):
        return {}


def save_progress(stars):
    try:
        with open(SAVE, 'w', encoding='utf8') as f:
            json.dump({'stars': stars}, f, indent=1)
    except OSError:
        pass


def unlocked(i, stars):
    return i == 0 or stars.get(MISSIONS[i - 1].key, 0) > 0


# ------------------------------------------------------------------ radio
# (speaker, text). Speakers: CONTROL (cyan), ROVER (white), SIGNAL (orange, the alien)
RADIO = {
    'start': ('CONTROL', 'Systems green, Pathfinder. The pad is {dist} tiles out. Go.'),
    'first_strike': ('SIGNAL', '... ... YOU  WILL  NOT  LEAVE ...'),
    'drop_seen': ('CONTROL', 'Boulder drop on your scope!'),
    'debris': ('CONTROL', 'Debris launch! Red rings mark the impact. Two turns.'),
    'debris_on_me': ('CONTROL', 'Incoming on your position. MOVE!'),
    'laser': ('ROVER', 'Laser fired. Path clear.'),
    'fused': ('ROVER', 'Fused rock ahead. The laser cannot cut it.'),
    'hit': ('CONTROL', 'You took a hit! Fuel down {dmg}.'),
    'shield': ('CONTROL', 'The shield took that one. It is spent.'),
    'scan': ('ROVER', 'Pulse complete. {found} boulders mapped.'),
    'low_fuel': ('CONTROL', 'Fuel is getting tight. No more detours.'),
    'no_route': ('CONTROL', 'Every route we know is blocked. Pulse, or blast through.'),
    'halfway': ('CONTROL', 'Halfway there. Keep it steady.'),
    'near_pad': ('CONTROL', 'We have you on the pad cameras. Bring it home!'),
    'quiet': ('CONTROL', 'The saucer has gone quiet. It is out of strikes.'),
    'win': ('CONTROL', 'Rover aboard! Shuttle lifting off!'),
    'lose': ('CONTROL', 'Pathfinder, do you read? ... Pathfinder ...'),
}
