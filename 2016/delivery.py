import numpy as np
import copy
from collections import defaultdict

def read_ints():
    return map(int, raw_input().split(' '))

class Warehouse:
    def __init__(self, location, products):
        # python tuple r, c
        self.location = location
        # Map product_id -> # products
        self.products = products
        
class Order:
    def __init__(self, location, items):
        # python tuple r, c
        self.location = location
        # Map product_id -> amount
        # To get all items as array: self.items.keys()
        # To get a key-value iterator: for k, v in self.items.iteritems()
        self.items = {}

if __name__ == '__main__':
    n_rows, n_cols, n_drones, n_turns, max_payload = read_ints()
    raw_input() # dont care
    # Every product type has an id which is the index to this array
    # The value at that index is its weight
    product_types = np.array(read_ints(), dtype=np.int32)
    n_warehouses = int(raw_input())
    warehouses = []

    for i in xrange(n_warehouses):
        location = tuple(read_ints())
        products = {}
        product_amounts = read_ints()
        for i, amount in enumerate(product_amounts):
            products[i] = amount
        warehouse = Warehouse(location, products)
        warehouses.append(warehouse)

    n_orders = int(raw_input())
    orders = []

    for i in xrange(n_orders):
        location = tuple(read_ints())
        n_items = int(raw_input())
        parsed_items = read_ints()
        items = defaultdict(0)
        for item in parsed_items:
            items[item] += 1
        order = Order(location, items)











