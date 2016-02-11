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
        self.center = center
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
            current = tuple(np.array(center) + direction)
            canvas[tuple(current)] = True
            # Check if diagonal, those have to paint most
            if not np.any(direction == 0):
                next_directions = [
                    direction, np.array([0, direction[1]]), np.array([direction[0], 0])
                ]
            else:
                next_directions = [direction]
            self.draw_help(canvas, current, radius-1, next_directions)

    def draw(self, canvas):
        canvas[tuple(self.center)] = True
        self.draw_help(canvas, self.center, self.radius)

    def __str__(self):
        return 'PAINT_SQUARE {} {} {}'.format(self.center[0], self.center[1], self.radius)

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
        assert (isinstance(self.cell, tuple))
        canvas[self.cell] = False

    def __str__(self):
        return 'ERASE_CELL {} {}'.format(self.cell[0], self.cell[1])

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

    def find_square_at(self, topleft, canvas, positive):
        holes = 0 if canvas[topleft] or positive[topleft] else 1
        fresh = 1 if positive[topleft] else 0
        last_fresh = fresh
        topleft = np.array(topleft)
        bottomright = copy.deepcopy(topleft)
        margin = np.min(np.array([n_rows, n_cols]) - topleft) - 1

        for step in range(2, margin):
            # every step, investigate growing the square. If that has too many holes, 
            # go with the smaller one
            for r in range(topleft[0], bottomright[0]+3):
                for c in range(bottomright[1]+1, bottomright[1]+3):
                    should_paint = positive[r,c]
                    already_painted = canvas[r,c]
                    if not (should_paint or already_painted):
                        holes += 1
                    elif should_paint:
                        fresh += 1
            for r in range(bottomright[0]+1, bottomright[0]+3):
                for c in range(topleft[1], bottomright[1]+1):
                    should_paint = positive[r,c]
                    already_painted = canvas[r,c]
                    if not (should_paint or already_painted):
                        holes += 1
                    elif should_paint:
                        fresh += 1

            # If more holes than the length of a side, don't bother with a square
            if holes > step + 1:
                break
                
            bottomright += np.array([2, 2])
            last_fresh = fresh

        radius = (bottomright - topleft) / 2
        center = topleft + radius
        radius = radius[0]
        return Square(center, radius), last_fresh

    def neighbors(self, goal):
        # 2. Try drawing the longest lines you can
        # 1. Try drawing some squares, taking into account there can be holes
        # 3. Try erasing cells that shouldn't be painted
        diff = goal.canvas - self.canvas
        negative = diff & self.canvas   # True where should be erased
        positive = diff & goal.canvas   # True where should still be painted

        squares = []
        # max_radius = -1
        max_fresh = 1 # Have at least one newly painted cell
        for r in range(n_rows):
            for c in range(n_cols):
                square, fresh = self.find_square_at((r, c), self.canvas, positive)
                if fresh > max_fresh:
                    max_fresh = fresh
                    squares = [square]
                elif fresh == max_fresh:
                    squares.append(square)

        # Let's be lazy and only find lines more than 1 cell long
        # single cells can be covered by squares
        max_length = 2
        lines = []
        current_start = None
        current_length = 0


        # Find horizontal lines. Keep only the longest
        # Don't care about lines with gaps
        for r in range(n_rows): # The +1 is so we can finish a row
            for c in range(n_cols + 1):
                # TODO add a rule for drawing over existing
                if c < n_cols and positive[r, c]:
                    if current_start is None:
                        current_start = (r, c)
                    current_length += 1
                elif current_length >= max_length:
                    if current_length > max_length:
                        max_length = current_length
                        lines = []
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

        for i in range(rs.size):
            r, c = (rs[i], cs[i])
            erases.append(Erase((r, c)))

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
            yield reconstruct_history(history, move_history, current)

        current_g = g[current]
        for neighbor, move in current.neighbors(goal):
            # The distance to a neighbor is always +1: a single move
            neighbor_g = current_g + 1
            
            # It's super picky so you'll never see equally good results, just
            # better ones
            if neighbor_g >= g.get(neighbor, float('inf')):
                continue    # Worse than previously known 
            if not neighbor in open_set:
                open_set.put(neighbor) # New route, add it
                
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

    # Program never stops, just keeps running trying to find improvements
    results = run_algo(canvas)
    for result in results:
        print len(result)
        for line in result:
            print line

