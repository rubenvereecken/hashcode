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

class Job:
    def __init__(self, warehouse, deliver_at):
        self.warehouse = warehouse
        self.products = {}
        self.deliver_at = deliver_at
        self.capacity = max_payload
        self.left = self.capacity

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

    def __contains__(self, item):
        return self.products[item] > 0

    def empty(self):
        return np.all(self.products.values() == 0)

    def contains_enough_of(self, item, amount):
        return self.products[item] >= amount

    def remove_products(self, item, amount):
        self.products[item] -= amount


class Order:
    def __init__(self, location=None, items=None, _id=None):
        # python tuple r, c
        self.location = location
        # Map product_id -> amount
        # To get all items as array: self.items.keys()
        # To get all values: self.items.values()
        # To get a key-value iterator: for k, v in self.items.iteritems()
        self.products = items
        self.id = _id
        self.jobs = []

    def create_jobs(self, warehouse, partial):
        jobs = []
        while not partial.empty():
            job = Job(warehouse, self.id)
            for product, amount_wanted in partial.products.iteritems():
                taking = min(amount_wanted, job.left // product_types[product])
                partial.products[product] -= taking
                if taking > 0:
                    job.products[product] = taking
                    job.left -= taking * product_types[product]
            assert (job.left != job.capacity)
            # if job.left != job.capacity:
            jobs.append(job)
        self.jobs += jobs

    def take_job(self):
        return self.jobs.pop()

    def has_jobs_left(self):
        return len(self.jobs) > 0

    def empty(self):
        return len(self.products) == 0 or np.all(np.array(self.products.values()) == 0)

    def total_weight(self):
        return sum(map(lambda (product, amount) : amount * product_types[product], self.products.iteritems()))

    def __str__(self):
        s = ''
        s += 'order for {}, contains {} product types, total of {} products'.format(self.location, len(self.products.keys()), sum(self.products.values()))
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
    products = defaultdict(int)
    for item in parsed_items:
        products[item] += 1
    order = Order(location, products, i)
    orders.append(order)


def euclid(a, b):
    if isinstance(a, tuple):
        a = np.array(a)
    if isinstance(b, tuple):
        b = np.array(b)
    return math.ceil(np.linalg.norm(b-a))

def determine_orders():
    # assume global
    drone_location = warehouses[0].location
    warehouses_to_drones = np.array(map(lambda warehouse: euclid(warehouse.location, drone_location), warehouses))
    orders_by_weight = sorted(orders, key=lambda order: order.total_weight())
    weighted_orders = []
    broken_warehouses = copy.deepcopy(warehouses)


    for order_idx, order in enumerate(orders_by_weight):
        # sorted by warehouse to order PLUS warehouse to drone start
        broken_warehouses.sort(key=lambda warehouse: euclid(warehouse.location, order.location) + euclid(warehouse.location, drone_location))
        min_travel = 0
        partials = []
        for warehouse_idx, warehouse in enumerate(broken_warehouses):
            if order.empty(): break
            partial = {}
            for product, amount_wanted in order.products.iteritems():
                warehouse_has = warehouse.products[product]
                # By taking the minimum of what the warehouse has to offer and
                # what you want, you can never take too much
                taking = min(warehouse_has, amount_wanted)
                if taking > 0:
                    partial[product] = taking
                    min_travel += (euclid(warehouse.location, order.location) + warehouses_to_drones[warehouse_idx])
                    order.products[product] -= taking
                    warehouse.products[product] -= taking
            partial_order = Order()
            partial_order.products = partial
            partials.append(partial_order)
            # TODO only create jobs if you're sure an order can be fulfilled

        # If managed to complete the order, create jobs for it
        # If order cannot be completed, don't even bother
        if order.empty():
            for partial_order in partials:
                order.create_jobs(warehouse, partial_order)
            weighted_orders.append((min_travel, order))
        else:
            for partial_order in partials:
                for prod in partial_order.products:
                    partial_order.warehouse.products[prod] += partial_order.products[prod]
                    order.products[prod] += partial_order.products[prod]

    weighted_orders.sort()
    # Extract only the orders, don't care for the weights
    return list(map(lambda x: x[1], weighted_orders))

# CALL THIS YE DRONES
def get_job():
    if len(orders) == 0:
        return None
    # current = next(map(lambda order: order.has_jobs_left()))
    current = orders[0]
    if not current.has_jobs_left():
        orders.pop(0)
        return get_job()
    return current.take_job()


class Drone(object):
    def __init__(self,_id):
        self.location = warehouses[0].location
        self.turnsLeft = 0
        self.commands = [] #lijst van strings
        self.id = _id
        self.payload = {

        }

    def calculateNewAction(self):
        global total_commands
        job = get_job()

        if job is None:
            return

        for product, amount in job.products.iteritems():
            total_commands += 1
            self.commands.append("{0} L {1} {2} {3}".format(self.id,job.warehouse.id, product, amount))

        for product, amount in job.products.iteritems():
            total_commands += 1
            self.commands.append("{0} D {1} {2} {3}".format(self.id, job.deliver_at, product, amount))

    def performTurn(self):
        if self.turnsLeft == 0:
            self.calculateNewAction()

        self.turnsLeft - 1;

for i in range(n_drones):
    drones.append(Drone(i));

order = determine_orders()

def main():
    for a in range(n_turns):
        for drone in drones:
            drone.performTurn()
    # output
    print(total_commands)
    for drone in drones:
        print("\n".join(drone.commands))


main()
