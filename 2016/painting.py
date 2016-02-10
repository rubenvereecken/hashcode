import copy
import numpy as np
from enum import Enum
from Queue import PriorityQueue

def read_ints():
    return map(int, raw_input().split(' '))

# Aww yis globals
n_rows = None
n_cols = None
goal = None

class Type(Enum):
    PAINT_SQUARE = 1
    PAINT_LINE = 2
    ERASE_CELL = 3

class Move():
    def draw(self, canvas):
        raise NotImplemented()

class Square(Move):
    def __init__(self, center, radius):
        # Square always has an odd edge length
        self.center = cell
        self.radius = radius

    def draw_help(self, canvas, center, radius, directions=None):
        if radius == 0:
            return
        center = np.array(center)
        if directions is None:
            directions = [
                np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]),
                np.array([0, 1]), np.array([1, 1]), np.array([1, 0]),
                np.array([1, -1]), np.array([0, -1])
            ]

        for direction in directions:
            current = center + direction
            canvas[current] = True
            # Check if diagonal, those have to paint most
            if not np.any(direction == 0):
                next_directions = [
                    direction, np.array([0, direction[1]]), np.array([direction[0], 0])
                ]
            else:
                next_directions = [direction]
            draw_help(canvas, current, radius-1, next_directions)


    def draw(self, canvas):
        canvas[self.center] = True
        draw_help(canvas, self.center, self.radius)


class Line(Move):
    def __init__(self, start, to):
        # assert(start[0] == to[0] or start[1] == to[1])
        self.start = start
        self.to = to
        
    def draw(self, canvas):
        if self.start[0] == self.to[0]: # horizontal
            r = self.start[0]
            for c in range(self.start[1], self.to[1]+1): # to is inclusive
                canvas[r,c] = True
        else:
            c = self.start[1]
            for r in range(self.start[0], self.to[0]+1):
                canvas[r,c] = True

    def __str__(self):
        return 'PAINT_LINE {} {} {} {}'.format(self.start[0], self.start[1],
                                               self.to[0], self.to[1])

class Erase(Move):
    def __init__(self, cell):
        self.cell = cell

    def draw(self, canvas):
        canvas[self.cell] = False

class State():
    def __init__(self, canvas=None):
        if not canvas is None:
            self.canvas = canvas
        else:
            self.canvas = np.zeros((n_rows, n_cols), dtype=np.bool)

    def copy(self):
        other = copy.deepcopy(self)
        return other

    def __eq__(self, other):
        return np.all(self.canvas == other.canvas)

    def __lt__(self, other):
        raise Exception()

    def __hash__(self):
        # Improvised 
        h = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if self.canvas[r,c]: h += r * n_cols + c
        return h

    def __str__(self):
        s = ''
        for r in range(n_rows):
            s += ''.join(map(lambda b: '#' if b else '.', self.canvas[r]))
            s += '\n'
        return s

    def heuristic(self, goal):
        # Heuristically and optimistically (so it's admissible) estimate the cost
        # from current state to goal
        inf = float('inf')
        min_r, min_c = (inf, inf)
        max_r, max_c = (-inf, -inf)
        diff = goal.canvas - self.canvas
        positive = diff & goal.canvas   # True where should be painted

        for r in range(n_rows):
            for c in range(n_cols):
                if positive[r, c]:
                    min_r = min(min_r, r)
                    min_c = min(min_c, c)
                    max_r = max(max_r, r)
                    max_c = max(max_c, c)
        if min_r == inf:
            # nothing left to paint, maybe erase but not paint
            # TODO room for improvement?
            return 0

        width = max_r - min_r + 1 # +1 because it's inclusive
        height = max_c - min_c + 1
        # This is a rectangle
        long_dim = max(width, height)
        short_dim = min(width, height)

        # Completely ignore erasing because the heuristic must be 
        # admissible, very important for A*
        # NEVER overestimate!
        # Suppose we would only draw lines to cover the surface
        min_lines = min(width, height)
        # Corresponds to throwing in as many squares as possible
        # and then filling up remaining space with lines
        square_scenario = long_dim // short_dim + (long_dim % short_dim)

        # underestimate underestimate underestimate
        return min(min_lines, square_scenario)

    def neighbors(self, goal):
        # 1. Try drawing the longest lines you can
        # 2. Try drawing some squares, taking into account there can be holes
        # 3. Try erasing cells that shouldn't be painted
        diff = goal.canvas - self.canvas
        negative = diff & self.canvas   # True where should be erased
        positive = diff & goal.canvas   # True where should be painted
        max_length = 1
        lines = []
        current_start = None
        current_length = 0

        # Find horizontal lines. Keep only the longest
        # Don't care about lines with gaps
        for r in range(n_rows): # The +1 is so we can finish a row
            for c in range(n_cols + 1):
                if c < n_cols and positive[r, c]:
                    if current_start is None:
                        current_start = (r, c)
                    current_length += 1
                elif current_length >= max_length:
                    if current_length > max_length:
                        max_length = current_length
                        lines = []
                    # print current_start, r, c-1
                    lines.append(Line(current_start, (r, c-1)))
                    current_start = None
                    current_length = 0
                if c >= n_cols:
                    current_start = None
                    current_length = 0

        # Find vertical lines
        for c in range(n_cols):
            for r in range(n_rows+1):
                if r < n_rows and positive[r, c]:
                    if current_start is None:
                        current_start = (r, c)
                    current_length += 1
                elif current_length >= max_length:
                    if current_length > max_length:
                        max_length = current_length
                        lines = []
                    lines.append(Line(current_start, (r-1, c)))
                    current_start = None
                    current_length = 0
                if r >= n_rows:
                    current_start = None
                    current_length = 0

        # Find cells to erase
        erases = []
        rs, cs = np.where(negative == True)
        for i in range(rs.length):
            r, c = (rs[i], cs[i])
            erases.append(Erase(r, c))

        squares = []
        # TODO squares

        # Keep erases for last
        moves = lines + squares + erases

        # Yield all neighbors along with the moves that generated them
        for move in moves:
            neighbor = self.copy()
            move.draw(neighbor.canvas)
            yield neighbor, move


class PrioritySet():
    def __init__(self, sort_key):
        # Need a pqueue with the ability to change priorities of items in the q
        # self.q = PriorityQueue()
        self.q = []
        self.set = set([])
        self.sort_key = sort_key

    def put(self, item):
        # self.q.put((priority, item))
        self.q.append(item)
        self.set.add(item)

    def get(self):
        # item = self.q.get_nowait()[]
        # This last minute sort is needed because items can have priorities
        # changed
        self.q.sort(key=self.sort_key) # Apply key to state
        item = self.q.pop(0)
        self.set.remove(item)
        return item

    def empty(self):
        return len(self.set) == 0

    def __contains__(self, item):
        return item in self.set

def reconstruct_history(history, move_history, current):
    path = []
    previous = history.get(current, None)
    previous_move = move_history.get(current, None)
    if previous is None:
        return []
    else:
        return reconstruct_history(history, move_history, previous) + [previous_move]

def run_algo(canvas):
    """
    A* 
    """
    start = State() # empty canvas
    goal = State(canvas) # self one is global
    history = {}
    move_history = {}

    g = {}
    g[start] = 0
    f = {}
    f[start] = start.heuristic(goal)

    open_set = PrioritySet(lambda x: f[x])
    open_set.put(start) # highest priority

    while not open_set.empty():
        current = open_set.get()
        if current == goal:
            # yield current.history
            # continue # Try finding more, better solutions
            return reconstruct_history(history, move_history, current)

        current_g = g[current]
        for neighbor, move in current.neighbors(goal):
            # TODO check if heuristic is consistent, otherwise use closed set
            # The distance to a neighbor is always +1: a single move
            neighbor_g = current_g + 1
            
            if not neighbor in open_set:
                open_set.put(neighbor) # New route, add it
            # TODO might be that it's not in open set but still better
            elif neighbor_g >= g[neighbor]:
                continue    # Worse than previously known 
                
            # Remember the best move and previous state to get here
            history[neighbor] = current
            move_history[neighbor] = move
            g[neighbor] = neighbor_g
            f[neighbor] = g[neighbor] + neighbor.heuristic(goal)




if __name__ == '__main__':
    n_rows, n_cols = read_ints()
    canvas = np.zeros((n_rows, n_cols), dtype=np.bool)

    for i in xrange(n_rows):
        line = raw_input()
        canvas[i, ...] = map(lambda c: True if c == '#' else False, line)

    result = run_algo(canvas)
    print len(result)
    for line in result:
        print line

