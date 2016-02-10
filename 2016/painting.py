import copy
import numpy as np
from enum import Enum
from Queue import PriorityQueue

def read_ints():
    return map(int, raw_input().split(' '))

# Aww yis globals
n_rows = None
n_cols = None

class Type(Enum):
    PAINT_SQUARE = 1
    PAINT_LINE = 2
    ERASE_CELL = 3

class Move():
    def __init__(self):
        pass

class Square(Move):
    def __init__(self, cell, size):
        self.cell = cell
        self.size = size

class Line(Move):
    def __init__(self, cell, size):
        self.cell = cell
        self.size = size

class Erase(Move):
    def __init__(self, cell):
        self.cell = cell

class State():
    # A crude way to make states hashable and referrable
    all_states = {}
    n_states = 0

    def __init__(self, canvas=None):
        self.canvas = canvas or np.zeros()
        State.register(self)

    def copy(self):
        other = copy.deepcopy(self)
        State.register(other)
        return other

    def __eq__(self, other):
        raise Exception()

    def __lt__(self, other):
        raise Exception()

    def __hash__(self):
        return self.id

    @classmethod
    def get(cls, id):
        return State.all_states[id]

    @classmethod
    def register(cls, state):
        state.id = cls.n_states
        cls.all_states[cls.n_states] = state
        cls.n_states += 1

    @classmethod
    def remove(cls, state):
        if isinstance(state, cls):
            state = state.id
        del all_states[state]

    # This is the toughest one, heuristically create neighbor states
    def neighbors(self):
        pass

    # Heuristically and optimistically (so it's admissible) estimate the cost
    # from current state to goal
    def heuristic(self):
        pass

class PrioritySet():
    def __init__(self):
        this.q = PriorityQueue()
        this.set = set([])

    def put(self, item, priority):
        this.q.put((priority, item))
        this.set.add(item)

    def get(self):
        item = this.q.get_nowait()[1]
        this.set.remove(item)
        return item

    def __contains__(self, item):
        return item in this.set


def run_algo(canvas):
    """
    A* 
    """
    start = State() # empty canvas
    goal = State(canvas)
    open_set = PriorityQueue()
    open_set.put(start, 0)

    g = {start.id: 0}
    f = heuristic(start)

    while not open_set.empty():
        current = open_set.get()
        if current == goal:
            return "whoop"

        current_g = g[current.id]
        for neighbor in current.neighbors():
            # TODO check if heuristic is consistent, otherwise use closed set
            # The distance to a neighbor is always +1: a single move
            neighbor_g = current_g + 1






if __name__ == '__main__':
    n_rows, n_cols = read_ints()
    canvas = np.zeros((n_rows, n_cols), dtype=np.bool)

    for i in xrange(n_rows):
        line = raw_input()
        canvas[i, ...] = map(lambda c: True if c == '#' else False, line)

    print canvas
