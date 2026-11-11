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

    def act(self, obs):
        raise NotImplementedError


def path_action(view, nxt):
    """Turn the next tile of a planned path into a rover action."""
    if nxt is None or nxt == view.rover:
        return ('wait', None)
    if view.belief.get(nxt, 0) == 1:  # SOFT
        return ('shoot', nxt)
    return ('move', nxt)
