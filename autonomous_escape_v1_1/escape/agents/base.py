class Agent:
    """Base class. Rovers get a rover_view() (fog applied); aliens get the
    full GameState and must not modify it."""
    name = 'agent'
    side = None

    def __init__(self):
        self.nodes = 0        # search nodes expanded (for the benchmark)
        self.info = ''        # one-line status shown in the HUD

    def reset(self):
        self.nodes = 0
        self.info = ''
        self._ties = None

    def tie_noise(self, gmap):
        """Tiny random extra cost per tile (far below one move), so that among
        equally cheap routes the rover picks a different one each game."""
        if getattr(self, '_ties', None) is None or self._ties[0] is not gmap:
            rng = getattr(self, 'rng', None)
            noise = {p: rng.random() * 0.35 for p in gmap.open_tiles} if rng else {}
            self._ties = (gmap, noise)
        return self._ties[1]

    def act(self, obs):
        raise NotImplementedError


def path_action(view, nxt):
    """Turn the next tile of a planned path into a rover action."""
    if nxt is None or nxt == view.rover:
        return ('wait', None)
    if view.belief.get(nxt, 0) == 1:  # SOFT
        return ('shoot', nxt)
    return ('move', nxt)
