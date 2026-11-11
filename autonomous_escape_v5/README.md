# Autonomous Escape — version 5 (docking run in deep space)

Built on version 4 (`../autonomous_escape_v4`). Same rules engine, maps and AIs, now in
open space above Mars:

* The rover is a shuttle (PATHFINDER-7): delta wings, three engines whose glow follows the
  thrust, RCS jets, banking into turns, a cockpit pilot view.
* Blocked tiles are clusters of tumbling asteroids; drifting rock and metal debris and
  space dust fill the space above and below the flight plane; a faint lattice marks the
  open lanes. Mars fills the view below, with nebula glow, Phobos and huge distant rocks.
* The goal is the docking port of the orbital station Ares Gate (spinning habitat ring,
  solar arrays, green guide lights); winning plays a docking sequence.
* The alien saucer stalks the shuttle, tows boulders into the lanes on its tractor beam
  and hurls debris that comes back as burning meteors; everything breaks apart weightless.

```
python main.py              # title screen
python main.py --mission 1  # straight into the campaign
```

---

# Autonomous Escape — version 3 (campaign, cockpit, cinematics)

Version 3 builds on the version 2 3D pilot view (`../autonomous_escape_v2`). The rules engine
and every AI are the same as versions 1 and 2, so the benchmark results still hold
(`python bench.py`). Version 3 adds:

* **Campaign** — five story missions on Phobos (prologue, briefings, mission control radio
  during play, an epilogue), each with its own map, alien AI, strikes and fuel, unlocked in
  order, rated with 1–3 stars and saved in `save.json`.
* **New rover systems** — Radar pulse (Q: maps every boulder within 10 tiles for 2 fuel) and
  a Debris shield (absorbs hits). They are rule parameters that are off by default, so the
  benchmark is unchanged.
* **Cockpit HUD** — compass tape with pad, debris and saucer bearings; a dashboard with a
  fuel gauge, a heading-up radar scope, alien strike pips and system status; radio messages;
  an orbital minimap; flashing INCOMING alerts.
* **Controls** — click the ground to send the autopilot there; right-drag to look around
  (pilot) or orbit (chase / overhead); mouse wheel zooms; Esc pause menu; H help.
* **Cinematics** — fly-in intro per mission, shuttle lift-off when you win, "signal lost"
  when you lose, then a debrief with stars and stats.
* **Sound** — synthesised laser, impacts, alarms, radio chirps, motor, launch and an ambient
  drone (generated once into `ui3d/sfx/`). M mutes.
* **3D title screen** — the menus float over a slow orbit of the debris field.

```
pip install -r requirements.txt
python main.py                 # title screen (campaign / free play / how to play)
python main.py --mission 3     # straight into campaign mission 3
python main.py --2d            # the original 2D game
```

---

## 3D pilot view (from version 2)

```
pip install -r requirements.txt      # adds panda3d
python main.py                       # 3D menu
python main.py --mode rover --camera Pilot --play
```

| Camera (V cycles) | What you see |
|---|---|
| Pilot | From the camera mast: boulder fields, crater pits, Mars on the horizon, debris streaking in |
| Chase | Behind and above the rover |
| Overhead | The whole field at an angle (default when you play the alien; click the ground to strike) |

Driving in Pilot / Chase: Up/W forward, Down/S reverse, Left/Right turn 90° (free, takes no
turn), Space wait. Drive into a warm, loose boulder to blast it. In Overhead the arrows
drive north/south/west/east. Tab hides the minimap (top right: the orbital map with your
route, radar, boulders and red debris rings).

The scene (`ui3d/`) is built from the same terrain data as the 2D board: a lit height-field
mesh, every rock of the 2D map as a faceted 3D rock, real sun shadows, stars, Mars filling
part of the sky, the alien's saucer with its tractor beam, glowing debris meteors with
landing rings, and the launch pad in its crater.

---

# Version 1 notes (2D)

An asteroid has hit the mining base on Phobos. The last rover must cross the grooved
debris field to the launch pad before a hostile alien seals every route.
A turn-based search game for the AI lab: rover AIs (A*, D* Lite, minimax, fuzzy logic)
against alien AIs (min-cut planning, MCTS).

## Setup

```
pip install -r requirements.txt
python main.py                 # menu
python main.py --mode alien --rover Minimax --play     # jump straight in
```

## Modes and controls

| Mode | You do |
|---|---|
| AI vs AI | Pick any rover and alien. Space pauses, → steps one move, + / − change speed |
| You are the Alien | Left-click: boulder on an open tile, or debris on a boulder. Right-click: debris on an open tile. Space: hold |
| You are the Rover | Arrows / WASD drive (hold to keep driving); bump a loose boulder to shoot it. Space waits. Boulders show only inside your radar |

Everywhere: F fog overlay, P rover route, C alien's min cut, G grid, R replay map, N new map, Esc menu.
The window can be resized or maximised.

## What you see

The board is a lit Phobos debris field generated for each map (`ui/terrain.py`),
styled after the concept art in the deck: cratered grey-brown regolith, impassable
boulder fields and deep crater pits, small drivable pebbles, and the launch pad in a
crater. The grid is hidden (G shows it). The rover knows the terrain from the orbital map;
the fog dims everything outside its radar and hides the alien's boulders there.
The route is drawn as a smooth beaded line. Incoming debris is drawn as a glowing streak with a red target ring and a turn
countdown. A banner warns when debris is about to land on the planned route.
The alien's saucer flies over to each strike.

## Rules (escape/config.py)

| Rule | Value |
|---|---|
| Map | 21×21 debris field: noise-grown rock clusters with 3 winding channels carved base → pad, 4-neighbour moves; crater floor around the pad |
| Fair start | No-build zone within 2 tiles of base and pad; 3–6 vertex-disjoint routes from the start tile (real choke points, never a single one) |
| Fuel | 2.0 × shortest route. Move or wait: 1. Laser shot on adjacent soft boulder: 2 (rover stays put) |
| Radar | Circle of radius 5 tiles. Ridges are known from orbit; boulders only when seen |
| Debris warning | Phobos has no air, so incoming debris is visible anywhere, and the rover sees where it lands |
| Alien | Acts every 3 rover turns, 8 strikes total, never on or next to the rover |
| Boulder | Lands now, soft (can be shot) |
| Debris | Lands 2 turns later: soft boulder → hard (laser-proof); open tile → soft boulder; on the rover → −6 fuel |
| Rover wins | Reaches the pad. Escape score = 100 × fuel left ÷ starting fuel |
| Alien wins | Fuel runs out, or no route the rover can still afford (including fully sealed) |

## Agents

| Rover | Idea |
|---|---|
| Flood Fill | BFS from the pad, treats boulders as walls. Baseline |
| A* | Weighted A* on its belief map (soft = shot + move, hard = wall), replans every turn |
| D* Lite | Same decisions as A*, incremental replanning (≈10× fewer node expansions) |
| Minimax | Alpha-beta, 4 rover plies plus the alien's strikes, on the belief map |
| Fuzzy | Mamdani controller (scikit-fuzzy): fuel margin, detour length, incoming debris → caution, which reweights A* |

| Alien | Idea |
|---|---|
| Random | Any legal tile |
| Greedy | Block or harden the next tile on the rover's cheapest route |
| Core | Sink-side minimum vertex cut (max-flow) + race test, greedy fallback |
| MCTS | UCT over alien strikes, greedy-rover model, block-the-route rollouts |

## Benchmark

```
python bench.py                                   # 5 rovers × 4 aliens × maps 100–299
python bench.py --seeds 100-139 --aliens Core,Greedy
```

Writes `results/games.csv`, `results/summary.csv` and `results/alien_win_heatmap.png`.
Win rates come with Wilson 95% intervals; every pairing plays the same maps.
Maps 0–99 were used for tuning, 100–299 are held out.

## Tests

```
python -m pytest
```

Rules (fuel, laser, debris timing, alien limits, end conditions), the search simulator
matching the real rules move by move, D* Lite matching A*, our max-flow matching
networkx, every agent pairing finishing, and headless UI checks.

## Layout

```
escape/config.py     rules and constants
escape/mapgen.py     debris-field generator, fairness check
escape/graph.py      A*, reverse Dijkstra, BFS, min vertex cut (Edmonds-Karp)
escape/state.py      GameState: the rules engine
escape/fastsim.py    lean copy of the rules for search
escape/agents/       rovers.py, aliens.py
escape/bench.py      tournament runner, Wilson CI, heatmap
ui/app.py            pygame-ce front end
ui/terrain.py        procedural Phobos ground (craters, boulders, hill shading, shadows)
ui/sprites.py        rover, boulders, saucer, base
ui/assets/space.jpg  background from the team deck
ui3d/app3d.py        3D game: menu, cameras, controls, HUD + minimap, effects
ui3d/world.py        3D scene: terrain mesh, rocks, sky, Mars, rover/saucer/boulder models
ui3d/meshes.py       procedural geometry and textures (numpy -> Panda3D)
ui3d/hud.py          cockpit HUD: compass, dashboard, radar scope, radio, minimap
ui3d/story.py        campaign missions, radio lines, save file
ui3d/audio.py        synthesised sound effects and ambient music
ui3d/widgets.py      2D UI kit (text, panels, buttons, stars)
```
