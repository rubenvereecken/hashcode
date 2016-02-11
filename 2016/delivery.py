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

    def total_weight(self):
        return sum(map(lambda (product, amount): amount * product_types[product], self.items.iteritems()))

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

for _ in range(n_drones):
    drones.push(Drone());

total_commands = 0

class Drone:
    def __init__(self):
        self.location = warehouses[0].location
        self.turnsLeft = 0
        self.commands = [] #lijst van strings
        self.payload = {

        }

    def calculateNewAction():
        min_dist = 10000000
        min_order = None
        for order in orders:
            dist = sqrt(self.location, order.location) #TODO
            if dist < min_dist:
                min_order = order
                min_dist = dist

        item_key = min_order.items.keys[0]
        min_order.items[item_key] -= 1
        if min_order.items[item_key] <= 0:
            min_order.items.remove(item_key)

        target_warehouse = None

        for warehouse in warehouses:
            if warehouse.products[item_key] > 0:
                target_warehouse = warehouse
                warehouse.products[item_key] -= 1
                total_commands += 2
                drones.commands.push("{0} L {1} {2} {3}".format(drone.id,warehouse.id,product_type,1))
                drones.commands.push("{0} D {1} {2} {3}".format(drone.id,min_order.id,product_type,1)
                self.turnsLeft = numpy.linalg.norm(numpy.array(drone.location) - numpy.array(warehouse.location)) + numpy.linalg.norm(numpy.array(warehouse.location) - numpy.array(min_order.location)) + 2
                break


    def performTurn():
        if self.turnsLeft = 0:
            calculateNewAction()
        self.turnsLeft - 1;

def main():
    for a in range(n_turns):
        for drone in drones:
            drone.performTurn()

    # output
    print(total_commands)
    for drone in drones:
        print(drone.commands.join("\n"));
