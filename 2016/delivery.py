import numpy as np
import copy
import math
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
drones = []
n_warehouses, n_orders = (None, None)
total_commands = 0

class Warehouse:
    def __init__(self, location, products, _id):
        # python tuple r, c
        self.location = location
        # Map product_id -> # products
        self.products = products
        self.id = _id
    def __str__(self):
        s = ''
        s += 'location {}, contains {} products\n'.format(self.location, len(self.products))
        s += str(self.products) + '\n'
        return s


class Order:
    def __init__(self, location, items, _id):
        # python tuple r, c
        self.location = location
        # Map product_id -> amount
        # To get all items as array: self.items.keys()
        # To get all values: self.items.values()
        # To get a key-value iterator: for k, v in self.items.iteritems()
        self.items = items
        self.id = _id
        self.done = False
        self.finalTurn = -1

    def total_weight(self):
        return sum(map(lambda product, amount : amount * product_types[product], self.items.iteritems()))

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
    products = defaultdict(int)
    product_amounts = read_ints()
    for k, amount in enumerate(product_amounts):
        products[k] = amount
    warehouse = Warehouse(location, products, i)
    warehouses.append(warehouse)

n_orders = int(raw_input())
for i in xrange(n_orders):
    location = tuple(read_ints())
    n_items = int(raw_input())
    parsed_items = read_ints()
    items = defaultdict(int)
    for item in parsed_items:
        items[item] += 1
    order = Order(location, items, i)
    orders.append(order)

def euclid(a, b):
    if isinstance(a, tuple):
        a = np.array(a)
    if isinstance(b, tuple):
        b = np.array(b)
    return math.ceil(np.linalg.norm(b-a))

def determine_warehouse_order():
    # assume global
    drone_location = warehouses[0].location
    warehouses_to_drones = np.array(map(lambda warehouse: euclid(warehouse.location, drone_location)), warehouses)

    # orders_by_weight = orders.sort(key=lambda order: )
    for order in orders:
        pass


class Drone(object):
    def __init__(self,_id):
        self.location = warehouses[0].location
        self.turnsLeft = 0
        self.commands = [] #lijst van strings
        self.id = _id
        self.payload = {

        }

    def calculateNewAction(self,turn):
        global total_commands

        min_dist = 10000000
        min_order = None
        for order in orders:
            if order.done:
                continue
            dist = euclid(self.location, order.location) #TODO
            if dist < min_dist:
                min_order = order
                min_dist = dist

        if min_order is None:
            return

        item_key = min_order.items.keys()[0]
        min_order.items[item_key] -= 1
        if min_order.items[item_key] <= 0:
            del min_order.items[item_key]
        if len(min_order.items.keys()) == 0:
            min_order.done = True

        target_warehouse = None

        for warehouse in warehouses:
            if warehouse.products[item_key] > 0:
                target_warehouse = warehouse
                warehouse.products[item_key] -= 1

                total_commands += 2
                self.commands.append("{0} L {1} {2} {3}".format(self.id,warehouse.id,item_key,1))
                self.commands.append("{0} D {1} {2} {3}".format(self.id,min_order.id,item_key,1))

                self.turnsLeft = euclid(self.location, warehouse.location) + euclid(warehouse.location, min_order.location) + 2
                min_order.finalTurn = max(order.finalTurn, self.turnsLeft + turn)
                break


    def performTurn(self,turn):
        if self.turnsLeft == 0:
            self.calculateNewAction(turn)

        self.turnsLeft -= 1;

for i in range(n_drones):
    drones.append(Drone(i));


def main():
    for turn in range(n_turns):
        for drone in drones:
            drone.performTurn(turn)

    score = 0
    for order in orders:
        #print(order.finalTurn)
        if (order.finalTurn != -1):
            score += (n_turns - order.finalTurn)/n_turns

    print("score:")
    print(score*100)

    # output
    print(total_commands)
    for drone in drones:
        print("\n".join(drone.commands))


main()
