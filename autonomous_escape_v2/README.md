# Autonomous Escape — version 2 (3D pilot view)

Version 2 plays the exact same game (same rules engine, maps and AIs as version 1 in
`../autonomous_escape`) in 3D, from the rover's pilot seat. The 2D game is still here:
`python main.py --2d`.

## 3D pilot view

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
```
