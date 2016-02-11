import numpy as np
import copy
from collections import defaultdict

def read_ints():
    return map(int, raw_input().split(' '))

# HOOP DECLARATIONS LEKKER GLOBAL
n_rows, n_cols, n_drones, n_turns, max_payload = (None, None, None, None, None)
# Every product type has an id which is the index to this array
# The value at that index is its weight
product_types = None
warehouses = []
orders = []
n_warehouses, n_orders = (None, None)

class Warehouse:
    def __init__(self, location, products):
        # python tuple r, c
        self.location = location
        # Map product_id -> # products
        self.products = products

    def __str__(self):
        s = ''
        s += 'location {}, contains {} products\n'.format(self.location, len(self.products))
        s += str(self.products) + '\n'
        return s
            
        
class Order:
    def __init__(self, location, items):
        # python tuple r, c
        self.location = location
        # Map product_id -> amount
        # To get all items as array: self.items.keys()
        # To get all values: self.items.values()
        # To get a key-value iterator: for k, v in self.items.iteritems()
        self.items = items

    def __str__(self):
        s = ''
        s += 'order for {}, contains {} product types, total of {} products'.format(self.location, len(self.items.keys()), sum(self.items.values()))
        return s

### MAIN - parse everything into globals ###
n_rows, n_cols, n_drones, n_turns, max_payload = read_ints()
raw_input() # dont care
product_types = np.array(read_ints(), dtype=np.int32)
warehouses = []
orders = []

n_warehouses = int(raw_input())
for i in xrange(n_warehouses):
    location = tuple(read_ints())
    products = {}
    product_amounts = read_ints()
    for i, amount in enumerate(product_amounts):
        products[i] = amount
    warehouse = Warehouse(location, products)
    warehouses.append(warehouse)

n_orders = int(raw_input())
for i in xrange(n_orders):
    location = tuple(read_ints())
    n_items = int(raw_input())
    parsed_items = read_ints()
    items = defaultdict(int)
    for item in parsed_items:
        items[item] += 1
    order = Order(location, items)
    orders.append(order)

# DOE DEES WEG 
for warehouse in warehouses: print warehouse
for order in orders: print order
