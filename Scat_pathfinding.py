import collections
import csv
import heapq
import geopy.distance
from intersection_name_pathfinding import data

# from main import search

# new_data = data

# creating a name tuple for storing data according to the index columns
Location_data = collections.namedtuple("location_data", "SCATS,Location,Latitude,Longitude")

data = {}
# reading in data using the csv file and storing it inside the dictionary using the name tuple with index marked columns
with open("template/SCAT.csv") as f:
    r = csv.DictReader(f)
    for d in r:
        s, n, x, y = str(d["SCATS"]), d["Location"], float(d["Latitude"]), float(d["Longitude"])
        data[s] = Location_data(s, n, x, y)


# using geo pandas to calculate distance between two scats passed to it while using dictionary to get corresponding
# coordinates
def get_distance(start, end):
    coord1 = (data[start].Latitude, data[start].Longitude)
    coord2 = (data[end].Latitude, data[end].Longitude)
    distance = (geopy.distance.geodesic(coord1, coord2)).km
    return distance


# defining a false distance to increase weight on intersection so that pathfinding algorithm doesn't consider it a
# neighbour
def heuristic(start, end, weight):
    if weight != "":
        distance = weight
    else:
        coord1 = (data[start].Latitude, data[start].Longitude)
        coord2 = (data[end].Latitude, data[end].Longitude)
        distance = (geopy.distance.geodesic(coord1, coord2)).km
    return distance


# getting the nearest neighbouring nodes using the distance between them
def get_neighbors(start_location, n):
    return sorted(data.values(), key=lambda x: get_distance(start_location, x.SCATS))[1:n + 1]


# tricking the algorithm to avoid node by returning false distance
def get_new_neighbors(start_location, n, weight):
    return sorted(data.values(), key=lambda x: heuristic(start_location, x.SCATS, weight))[1:n + 1]


# making the previous node the parent node for every next node
def get_parent_node(closed_list, index):
    path = []
    while index is not None:
        path.append(index)
        index = closed_list.get(index, None)
    return [data[s] for s in path[::-1]]


# main function for A* path finding to get first route
def get_first(origin, destination):
    # pass in the available nodes here need to refactor
    Node = collections.namedtuple("Node", "SCATS,F,G,H,parent_SCATS")

    h = get_distance(origin, destination)
    open_list = [(h, Node(origin, h, 0, h, None))]  # heap
    closed_list = {}  # maps visited nodes to parent

    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    while len(open_list) >= 1:
        _, current_Location = heapq.heappop(open_list)
        print(current_Location)

        # check if current location exists in closed path list
        if current_Location.SCATS in closed_list:
            continue
        closed_list[current_Location.SCATS] = current_Location.parent_SCATS
        if current_Location.SCATS == destination:
            print("Complete route 1")
            # get all nodes and append them to empty lists
            for o in get_parent_node(closed_list, current_Location.SCATS):
                SCAT.append(o.SCATS)
                Location.append(o.Location)
                Latitude.append(o.Latitude)
                Longitude.append(o.Longitude)
            # print(SCAT, Location, Latitude, Longitude)

            break
        # loop for finding the next neighbours and storing them using heapq
        for neighbour in get_neighbors(current_Location.SCATS, 9):
            g = current_Location.G + get_distance(current_Location.SCATS, neighbour.SCATS)
            h = get_distance(neighbour.SCATS, destination)
            f = g + h
            heapq.heappush(open_list, (f, Node(neighbour.SCATS, f, g, h, current_Location.SCATS)))

    return SCAT, Location, Latitude, Longitude


# modified A* path finding to avoid certain paths.
def get_second(origin, destination):
    # pass in the available nodes here need to refactor
    Node = collections.namedtuple("Node", "SCATS,F,G,H,parent_SCATS")

    h = get_distance(origin, destination)
    open_list = [(h, Node(origin, h, 0, h, None))]  # heap
    closed_list = {}  # maps visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    # was suppose to get data from machine learning here in this list.
    paths_to_avoid = ['4040', '3180', '4057', '3682', '2825', '3804', '4034']
    while len(open_list) >= 1:
        _, current_Location = heapq.heappop(open_list)
        print(current_Location)

        # check if current location exists in closed path list
        if current_Location.SCATS in closed_list:
            continue
        closed_list[current_Location.SCATS] = current_Location.parent_SCATS
        if current_Location.SCATS == destination:
            print("Complete route 2")
            # get all nodes and append them to empty lists
            for o in get_parent_node(closed_list, current_Location.SCATS):
                SCAT.append(o.SCATS)
                Location.append(o.Location)
                Latitude.append(o.Latitude)
                Longitude.append(o.Longitude)
                # print(SCAT, Location, Latitude, Longitude)

            break

        # loop for finding the next neighbours and storing them using heapq
        for neighbours in get_neighbors(current_Location.SCATS, 9):
            # was supposed to get data from machine learning model to compare the value so that it can increase path
            # on that intersection
            if neighbours.SCATS in paths_to_avoid:
                weight = 7
                for next in get_new_neighbors(neighbours.SCATS, 2, weight):
                    g = current_Location.G + get_distance(neighbours.SCATS, next.SCATS)
                    h = get_distance(next.SCATS, destination)
                    f = g + h
                    heapq.heappush(open_list, (f, Node(next.SCATS, f, g, h, current_Location.SCATS)))
            else:
                g = current_Location.G + get_distance(current_Location.SCATS, neighbours.SCATS)
                h = get_distance(neighbours.SCATS, destination)
                f = g + h
                heapq.heappush(open_list, (f, Node(neighbours.SCATS, f, g, h, current_Location.SCATS)))

    return SCAT, Location, Latitude, Longitude


# modified A* path finding to avoid certain paths.
def get_third(origin, destination, avoid_scats):
    # pass in the available nodes here need to refactor
    Node = collections.namedtuple("Node", "SCATS,F,G,H,parent_SCATS")

    h = get_distance(origin, destination)
    open_list = [(h, Node(origin, h, 0, h, None))]  # heap
    closed_list = {}  # maps visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    while len(open_list) >= 1:
        _, current_Location = heapq.heappop(open_list)
        print(current_Location)
        # check if current location exists in closed path list
        if current_Location.SCATS in closed_list:
            continue
        closed_list[current_Location.SCATS] = current_Location.parent_SCATS
        if current_Location.SCATS == destination:
            print("Complete route 3")
            # get all nodes and append them to empty lists
            for o in get_parent_node(closed_list, current_Location.SCATS):
                SCAT.append(o.SCATS)
                Location.append(o.Location)
                Latitude.append(o.Latitude)
                Longitude.append(o.Longitude)
                # print(SCAT, Location, Latitude, Longitude)

            break

        # loop for finding the next neighbours and storing them using heapq
        for neighbours in get_neighbors(current_Location.SCATS, 7):
            # was supposed to get data from machine learning model to compare the value so that it can increase path
            # on that intersection
            if neighbours.SCATS in avoid_scats:
                weight = 5
                for next in get_new_neighbors(neighbours.SCATS, 2, weight):
                    g = current_Location.G + get_distance(neighbours.SCATS, next.SCATS)
                    h = get_distance(next.SCATS, destination)
                    f = g + h
                    heapq.heappush(open_list, (f, Node(next.SCATS, f, g, h, current_Location.SCATS)))
            else:
                g = current_Location.G + get_distance(current_Location.SCATS, neighbours.SCATS)
                h = get_distance(neighbours.SCATS, destination)
                f = g + h
                heapq.heappush(open_list, (f, Node(neighbours.SCATS, f, g, h, current_Location.SCATS)))

    return SCAT, Location, Latitude, Longitude


''' cannot use this anymore since machine learning uses intersection name and not scat number
# modified A* path finding to avoid certain paths.
def get_fourth(origin, destination):
    # pass in the available nodes here need to refactor
    Node = collections.namedtuple("Node", "SCATSLocation,F,G,H,parent_SCATS")

    h = get_distance(origin, destination)
    open_list = [(h, Node(origin, h, 0, h, None))]  # heap
    closed_list = {}  # maps visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    # data = get_loc(origin, destination)
    # SC, Loc, Lat, Long = data
    # Loc = data[n]
    new = list(new_data.keys())
    print(new)
    backend = search(new)
    # print(backend)
    while len(open_list) >= 1:
        _, current_Location = heapq.heappop(open_list)
        print(current_Location)
        # check if current location exists in closed path list
        if current_Location.SCATSLocation in closed_list:
            continue
        closed_list[current_Location.SCATSLocation] = current_Location.parent_SCATS
        if current_Location.SCATSLocation == destination:
            print("Complete")
            # get all nodes and append them to empty lists
            for o in get_parent_node(closed_list, current_Location.SCATSLocation):
                SCAT.append(o.SCATS)
                Location.append(o.Location)
                Latitude.append(o.Latitude)
                Longitude.append(o.Longitude)
                # print(SCAT, Location, Latitude, Longitude)

            break

        # loop for finding the next neighbours and storing them using heapq
        for neighbours in get_neighbors(current_Location.SCATSLocation, 7):
            # was supposed to get data from machine learning model to compare the value so that it can increase path
            # on that intersection
            if neighbours.Location in backend:
                if backend[current_Location.SCATSLocation] >= 170:
                    weight = 8
                    for next in get_new_neighbors(neighbours.SCATS, 2, weight):
                        g = current_Location.G + get_distance(neighbours.SCATS, next.SCATS)
                        h = get_distance(next.SCATS, destination)
                        f = g + h
                        heapq.heappush(open_list, (f, Node(next.SCATS, f, g, h, current_Location.SCATSLocation)))
            else:
                g = current_Location.G + get_distance(current_Location.SCATSLocation, neighbours.SCATS)
                h = get_distance(neighbours.SCATS, destination)
                f = g + h
                heapq.heappush(open_list, (f, Node(neighbours.SCATS, f, g, h, current_Location.SCATSLocation)))

    return SCAT, Location, Latitude, Longitude
'''

# some tests below to consider
'''
origin_scat = '0970'
destination_scat = '4321'

path = get_first(origin_scat,destination_scat)
S, N, L, l = path

for i in range(len(S)):
    print(S[i], N[i], L[i], l[i])

path = get_second(origin_scat,destination_scat)
S, N, L, l = path

for i in range(len(S)):
    print(S[i], N[i], L[i], l[i])

avoid_scats = ['4040', '3180', '4057', '3682', '2825', '3804', '4034', '3662', '3812']
path = get_third(origin_scat, destination_scat, avoid_scats)
S, N, L, l = path

for i in range(len(S)):
    print(S[i], N[i], L[i], l[i])

path = get_fourth(origin_scat, destination_scat)
S, N, L, l = path

for i in range(len(S)):
    print(S[i], N[i], L[i], l[i])
'''
