import collections
import csv
import heapq
import time

import geopy.distance
from main import search

# creating a name tuple for storing data according to the index columns
Location_data = collections.namedtuple("location_data", "SCATS,Location,Latitude,Longitude")

data = {}
# reading in data using the csv file and storing it inside the dictionary using the name tuple with index marked columns
with open("template/SCAT.csv") as f:
    r = csv.DictReader(f)
    for d in r:
        s, n, x, y = str(d["SCATS"]), str(d["Location"]), float(d["Latitude"]), float(d["Longitude"])
        data[n] = Location_data(s, n, x, y)


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
    return sorted(data.values(), key=lambda x: get_distance(start_location, x.Location))[1:n + 1]


# tricking the algorithm to avoid node by returning false distance
def get_new_neighbors(start_location, n, weight):
    return sorted(data.values(), key=lambda x: heuristic(start_location, x.Location, weight))[1:n + 1]


# making the previous node the parent node for every next node
def get_parent_node(closed_list, index):
    path = []
    while index is not None:
        path.append(index)
        index = closed_list.get(index, None)
    return [data[n] for n in path[::-1]]


# main function for A* path finding to get first route
def get_loc(origin, destination):
    # pass in the available nodes here need to refactor
    Node = collections.namedtuple("Node", "Location,F,G,H,parent_Location")

    h = get_distance(origin, destination)
    open_list = [(h, Node(origin, h, 0, h, None))]  # heap
    closed_list = {}  # maps visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    # loop until end of open path list reached
    while len(open_list) >= 1:
        _, current_Location = heapq.heappop(open_list)
        print(current_Location)

        # check if current location exists in closed path list
        if current_Location.Location in closed_list:
            continue
        closed_list[current_Location.Location] = current_Location.parent_Location
        if current_Location.Location == destination:
            print("Complete route")
            # get all nodes and append them to empty lists
            for o in get_parent_node(closed_list, current_Location.Location):
                SCAT.append(o.SCATS)
                Location.append(o.Location)
                Latitude.append(o.Latitude)
                Longitude.append(o.Longitude)
                # print(SCAT, Location, Latitude, Longitude)

            break
        # loop for finding the next neighbours and storing them using heapq
        for neighbour in get_neighbors(current_Location.Location, 9):
            g = current_Location.G + get_distance(current_Location.Location, neighbour.Location)
            h = get_distance(neighbour.Location, destination)
            f = g + h
            heapq.heappush(open_list, (f, Node(neighbour.Location, f, g, h, current_Location.Location)))

    return SCAT, Location, Latitude, Longitude


# modified A* path finding to avoid certain paths.
def get_new_loc(origin, destination):
    # pass in the available nodes here need to refactor
    Node = collections.namedtuple("Node", "Location,F,G,H,parent_Location")

    h = get_distance(origin, destination)
    open_list = [(h, Node(origin, h, 0, h, None))]  # heap
    closed_list = {}  # maps visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    paths_to_avoid = ['RIVERSDALE_RD W of BURKE_RD', 'BALWYN_RD S of DONCASTER_RD', 'BALWYN_RD N OF BELMORE_RD',
                      'WARRIGAL_RD S OF RIVERSDALE_RD', 'BURKE_RD S of EASTERN_FWY', 'TRAFALGAR_RD S of RIVERSDALE_RD',
                      'WHITEHORSE_RD E OF BURKE_RD']
    # loop until end of open path list reached
    while len(open_list) >= 1:
        _, current_Location = heapq.heappop(open_list)
        print(current_Location)

        # check if current location exists in closed path list
        if current_Location.Location in closed_list:
            continue
        closed_list[current_Location.Location] = current_Location.parent_Location
        if current_Location.Location == destination:
            print("Complete route 2")
            # get all nodes and append them to empty lists
            for o in get_parent_node(closed_list, current_Location.Location):
                SCAT.append(o.SCATS)
                Location.append(o.Location)
                Latitude.append(o.Latitude)
                Longitude.append(o.Longitude)
                # print(SCAT, Location, Latitude, Longitude)

            break
        # loop for finding the next neighbours and storing them using heapq while also checking if node discovered
        # exists in paths to avoid if yes increase the wight on that path and find another
        for neighbour in get_neighbors(current_Location.Location, 7):
            if neighbour.Location in paths_to_avoid:
                weight = 6
                for next in get_new_neighbors(neighbour.Location, 2, weight):
                    g = current_Location.G + get_distance(neighbour.Location, next.Location)
                    h = get_distance(next.Location, destination)
                    f = g + h
                    heapq.heappush(open_list, (f, Node(next.Location, f, g, h, current_Location.Location)))
            else:
                g = current_Location.G + get_distance(current_Location.Location, neighbour.Location)
                h = get_distance(neighbour.Location, destination)
                f = g + h
                heapq.heappush(open_list, (f, Node(neighbour.Location, f, g, h, current_Location.Location)))

    return SCAT, Location, Latitude, Longitude


def get_third_loc(origin, destination):
    # pass in the available nodes here need to refactor
    Node = collections.namedtuple("Node", "Location,F,G,H,parent_Location")

    h = get_distance(origin, destination)
    open_list = [(h, Node(origin, h, 0, h, None))]  # heap
    closed_list = {}  # maps visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    # data = get_loc(origin, destination)
    # SC, Loc, Lat, Long = data
    #Loc = data[n]

    new = list(data.keys())
    #print(new)
    backend = search(new)
    #time.sleep(300)
    #print(backend)


    # loop until end of open path list reached
    while len(open_list) >= 1:
        _, current_Location = heapq.heappop(open_list)
        print(current_Location)

        # check if current location exists in closed path list
        if current_Location.Location in closed_list:
            continue
        closed_list[current_Location.Location] = current_Location.parent_Location
        if current_Location.Location == destination:
            print("Complete route 3")
            # get all nodes and append them to empty lists
            for o in get_parent_node(closed_list, current_Location.Location):
                SCAT.append(o.SCATS)
                Location.append(o.Location)
                Latitude.append(o.Latitude)
                Longitude.append(o.Longitude)
                # print(SCAT, Location, Latitude, Longitude)

            break
        # loop for finding the next neighbours and storing them using heapq while also checking if node discovered
        # consider its traffic value using the referencing data obtained from the machine learning model if yes
        # increase the wight on that path and find another while ignoring the close by nodes by increasing its weight to a higher value
        for neighbour in get_neighbors(current_Location.Location, 7):
            if neighbour.Location in backend:
                if backend[current_Location.Location] >= 170:
                    weight = 9
                    for next in get_new_neighbors(neighbour.Location, 2, weight):
                        g = current_Location.G + get_distance(neighbour.Location, next.Location)
                        h = get_distance(next.Location, destination)
                        f = g + h
                        heapq.heappush(open_list, (f, Node(next.Location, f, g, h, current_Location.Location)))
            else:
                g = current_Location.G + get_distance(current_Location.Location, neighbour.Location)
                h = get_distance(neighbour.Location, destination)
                f = g + h
                heapq.heappush(open_list, (f, Node(neighbour.Location, f, g, h, current_Location.Location)))
    return SCAT, Location, Latitude, Longitude


def get_fourth_loc(origin, destination):
    # pass in the available nodes here need to refactor
    Node = collections.namedtuple("Node", "Location,F,G,H,parent_Location")

    h = get_distance(origin, destination)
    open_list = [(h, Node(origin, h, 0, h, None))]  # heap
    closed_list = {}  # maps visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []

    # data = get_loc(origin, destination)
    # SC, Loc, Lat, Long = data
    Loc = data[n]
    new = list(data.keys())
    # print(new)
    backend = search(new)
    #time.sleep(300)
    # print(backend)
    # loop until end of open path list reached
    while len(open_list) >= 1:
        _, current_Location = heapq.heappop(open_list)
        print(current_Location)

        # check if current location exists in closed path list
        if current_Location.Location in closed_list:
            continue
        closed_list[current_Location.Location] = current_Location.parent_Location
        if current_Location.Location == destination:
            print("Complete route 4")
            # get all nodes and append them to empty lists
            for o in get_parent_node(closed_list, current_Location.Location):
                SCAT.append(o.SCATS)
                Location.append(o.Location)
                Latitude.append(o.Latitude)
                Longitude.append(o.Longitude)
                # print(SCAT, Location, Latitude, Longitude)

            break
        # loop for finding the next neighbours and storing them using heapq while also checking if node discovered
        # consider its traffic value using the referencing data obtained from the machine learning model if yes
        # increase the wight '5' on that path and find another  while considering the close by nodes
        for neighbour in get_neighbors(current_Location.Location, 7):
            if neighbour.Location in backend:
                if backend[current_Location.Location] >= 110:
                    weight = 4
                    for next in get_new_neighbors(neighbour.Location, 2, weight):
                        g = current_Location.G + get_distance(neighbour.Location, next.Location)
                        h = get_distance(next.Location, destination)
                        f = g + h
                        heapq.heappush(open_list, (f, Node(next.Location, f, g, h, current_Location.Location)))
            else:
                g = current_Location.G + get_distance(current_Location.Location, neighbour.Location)
                h = get_distance(neighbour.Location, destination)
                f = g + h
                heapq.heappush(open_list, (f, Node(neighbour.Location, f, g, h, current_Location.Location)))

    return SCAT, Location, Latitude, Longitude


# some Tests below to consider
'''
origin_name = 'HIGH STREET_RD E of WARRIGAL_RD'
destination_name = 'HARP_RD E OF HIGH_ST'

raw = get_loc(origin_name,destination_name)
SC2, Loc2, Lat2, Long2 = raw

for i in range(len(Loc2)):
    print(SC2[i], Loc2[i], Lat2[i], Long2[i])

raw = get_new_loc(origin_name,destination_name)
SC2, Loc2, Lat2, Long2 = raw

for i in range(len(Loc2)):
    print(SC2[i], Loc2[i], Lat2[i], Long2[i])

raw = get_third_loc(origin_name, destination_name)
SC2, Loc2, Lat2, Long2 = raw

for i in range(len(Loc2)):
    print(SC2[i], Loc2[i], Lat2[i], Long2[i])

raw = get_fourth_loc(origin_name,destination_name)
SC2, Loc2, Lat2, Long2 = raw

for i in range(len(Loc2)):
    print(SC2[i], Loc2[i], Lat2[i], Long2[i])
'''