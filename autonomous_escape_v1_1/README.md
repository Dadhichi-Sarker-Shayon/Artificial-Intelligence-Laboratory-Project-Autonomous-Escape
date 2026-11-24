# Autonomous Escape — version 1.1 (random debris fields)

The final 2D game (version 1), where every game is new and nothing is fixed.

* **A new debris field every game** (`escape/mapgen.py`, `generate_random`):
  * random size from 41 x 41 to 61 x 61, with the base and pad on random opposite sides or corners
  * clustered boulders and crater pits with winding channels carved between base and pad,
    plus one or two random features: crater-rim rings, walls of rocks, diagonal ridges,
    rock islands, or extra boulder scatter
  * accepted only with at least 3 independent routes (fair) and routes that bend around
    the debris.
* **Rules drawn every game** (`escape/config.py`, `random_rules`): how often the alien
  strikes (every 3-7 turns), its strike budget (sized to the map's narrowest choke point,
  around what a full seal needs), starting fuel (1.6-2.2x the shortest route), plus
  random strike timing, debris delay (1-3 turns) and debris damage (4-8).
* **Unpredictable AIs:** rovers choose between nearly-equal routes at random; the Core alien
  varies which choke tiles it hits, how bold it is, and sometimes harasses instead of
  sealing; Greedy picks one of its top three targets.
* **Five regions** set the look only (colours, sun, signature feature): Stickney Rim
  (iron-red, giant crater), Kepler Grooves (basalt, grooves), Limtoc Basin (pale mesas),
  Reldresal Maze (brown, boulders), Drunlo Serpentine (blue-grey frost). All draw the
  debris as boulder clusters and crater pits.
* **Replay:** each game's seed is shown on screen ("#..."); `GameState.random_game(seed)`
  rebuilds exactly that game.
* **Camera:** opens zoomed out to fit the map; wheel zooms at the cursor, right- or middle-drag
  pans (middle only when playing the alien), Z shows the whole map, V follows the rover.
* **Balance** over 48 random games: Core alien beats A* 6/16 and Fuzzy 2/16; Greedy 0/16.

---|---|---|---|
  | Stickney Rim | concentric crater rims with gaps | iron-red dust, giant crater with rays | 12 strikes |
  | Kepler Grooves | long horizontal walls of pits, staggered gaps | dark basalt, parallel grooves | 10 strikes |
  | Limtoc Basin | open plains with big mesa islands | pale dust, smooth plains | 14 strikes |
  | Reldresal Maze | dense scatter of small boulders | brown boulder fields | 12 strikes |
  | Drunlo Serpentine | zigzag diagonal ridges (switchbacks) | blue-grey frost, ice ridges | 10 strikes |

  Every layout still guarantees at least 5 independent routes. Styles live in
  `escape/config.py` (`STYLE_RULES`) and `escape/mapgen.py`; looks in `ui/terrain.py` (`THEMES`).
* **Camera:** a map no longer fits the screen at full size, so the game opens zoomed out to
  show the whole map. Mouse wheel zooms at the cursor, right- or middle-drag pans (middle
  only when playing the alien), Z shows the whole map, V follows the rover, [ and ] zoom.
* **Every game is random.** Each game draws a fresh seed (shown on screen as "game #...").
  The alien's strike timing varies by up to half its interval, its budget by one strike, debris lands
  1-3 turns after launch and hits for 4-8 fuel, the Random and MCTS aliens use the game seed,
  and rovers pick randomly between equally good routes. A game seed is recorded with every
  benchmark row, so any game can be replayed exactly (`play(map, rover, alien, game_seed=...)`).
  Set `random_events=False` in `escape/config.py` for the old fixed timings.
* **Balance:** the alien strikes about every 5 turns on all maps. Core alien vs A* rover over
  6 random games per map: Stickney Rim 2/6, Kepler Grooves 3/6, Limtoc Basin 2/6,
  Reldresal Maze 3/6, Drunlo Serpentine 4/6 alien wins. Greedy never won.
* Terrain is generated fresh for every game (2-5 seconds behind a loading screen).

---

# Autonomous Escape

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
```
