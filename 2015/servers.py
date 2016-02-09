from copy import deepcopy
import numpy as np

def ils(create_initial, local_search, finished, perturb, better):
    old_solution = None
    current_solution = create_initial()
    print current_solution
    best_score = local_search(current_solution)

    while not finished():
        old_solution = deepcopy(current_solution)
        perturb(current_solution)
        best_score = local_search(current_solution, picker, neighborhood)
        current_solution = better(old_solution, current_solution)
        best_score = max(score(current_solution), best_score)
        

    
def read_ints():
    return map(int, raw_input().split(' '))


if __name__ == '__main__':
    n_rows, n_slots, n_unavailable, n_pools, n_servers = read_ints() 
    print 'n_pools', n_pools
    available = np.ones((n_rows, n_slots), dtype=np.bool)
    taken = np.zeros_like(available)
    pools = [[] for _ in xrange(n_pools)]
    servers = []
    reserve = []

    for i in xrange(n_unavailable):
        row, slot = read_ints()
        available[row, slot] = False

    for i in xrange(n_servers):
        size, capacity = read_ints()
        servers.append((size, capacity))

    servers.sort(reverse=True)

    def mark_taken(row, start, server, pool):
        size = server[0]
        taken[row, start:start+size] = True
        print pool
        pools[pool].append(server)
        pass

    def fits(row, start, server):
        size = server[0]
        print 'fits', server, 'at', (row, start)
        fits_size_wise = start + size <= n_slots
        return fits_size_wise and np.all(available[row,start:start+size]) \
                              and not np.any(taken[row,start:start+size])


    def create_initial():
        row, slot = (0, 0)
        for_later = []
        pool_i = 0
        for i, server in enumerate(servers):
            try:
                idx, (size, (row, slot)) = next((scrap for scrap in enumerate(for_later) if scrap[1][0] <= server[0]))
                mark_taken(row, slot, server, pool_i)
                pool_i = (pool_i + 1) % n_pools
                for_later.pop(idx)
                # Leftovuurrss
                if size > server[0]:
                    for_later.append((size-server[0], (row, slot+server[0])))
            except:
                # Ok np mate
                pass
            # Skip the occupied ones
            while not available[row][slot]:
                if slot < n_slots - 1:
                    slot += 1
                else:
                    row += 1
                    slot = 0
            while True:
                if row >= n_rows:
                    print taken
                    raise Exception("You've gone too far")
                if fits(row, slot, server):
                    mark_taken(row, slot, server, pool_i)
                    pool_i = (pool_i + 1) % n_pools
                    break
                else:
                    # exclusive
                    until = n_slots
                    for x in range(slot+1, n_slots):
                        if not available[row, x] or taken[row, x]:
                            until = x 
                    for_later.append((until-slot, (row, slot)))
                    
                    if until >= n_slots - 1:
                        row += 1
                        slot = 0
                    else:
                        slot = until + 1
                        while not available[row][slot]:
                            if slot < n_slots - 1:
                                slot += 1
                            else:
                                row += 1
                                slot = 0
        solution = (taken, pools)
        return solution

    ils(create_initial, None, None, None, None)




