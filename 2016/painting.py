import copy
import numpy as np

def read_ints():
    return map(int, raw_input().split(' '))


if __name__ == '__main__':
    n_rows, n_cols = read_ints()
    canvas = np.zeros((n_rows, n_cols), dtype=np.bool)

    for i in xrange(n_rows):
        line = raw_input()
        canvas[i, ...] = map(lambda c: True if c == '#' else False, line)

    print canvas
