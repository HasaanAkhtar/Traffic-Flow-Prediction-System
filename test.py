# to test the path finding run this file to see the path creation using the backend tensor flow.
import collections
import csv
import heapq
from main import search
import geopy.distance

Location_data = collections.namedtuple("location_data", "SCATS,Location,Latitude,Longitude")

data = {}
with open("template/SCAT.csv") as f:
    r = csv.DictReader(f)
    for d in r:
        s, n, x, y = str(d["SCATS"]), d["Location"], float(d["Latitude"]), float(d["Longitude"])
        data[s] = Location_data(s, n, x, y)


def getdistance(start, end):
    coord1 = (data[start].Latitude, data[start].Longitude)
    coord2 = (data[end].Latitude, data[end].Longitude)
    distance = (geopy.distance.geodesic(coord1, coord2)).km
    return distance


def heuristic(start, end, weight):
    if weight != "":
        distance = weight
    else:
        coord1 = (data[start].Latitude, data[start].Longitude)
        coord2 = (data[end].Latitude, data[end].Longitude)
        distance = (geopy.distance.geodesic(coord1, coord2)).km
    return distance


def getneighbors(startlocation, n=10):
    return sorted(data.values(), key=lambda x: getdistance(startlocation, x.SCATS))[1:n + 1]


def getnewneighbors(startlocation, n, weight):
    return sorted(data.values(), key=lambda x: heuristic(startlocation, x.SCATS, weight))[1:n + 1]


def getParent(closedlist, index):
    path = []
    while index is not None:
        path.append(index)
        index = closedlist.get(index, None)
    return [data[s] for s in path[::-1]]


def getfirst(o, d):
    startIndex = o
    endIndex = d

    # pass in the available nodes here need to refractor
    Node = collections.namedtuple("Node", "SCATS,F,G,H,parentSCATS")

    h = getdistance(startIndex, endIndex)
    openlist = [(h, Node(startIndex, h, 0, h, None))]  # heap
    closedlist = {}  # map visited nodes to parent

    SCAT = []
    Location = []
    Latitude = []
    Longitude = []

    while len(openlist) >= 1:
        _, currentLocation = heapq.heappop(openlist)
        print(currentLocation)

        if currentLocation.SCATS in closedlist:
            continue
        closedlist[currentLocation.SCATS] = currentLocation.parentSCATS
        if currentLocation.SCATS == endIndex:
            print("Complete")
            for p in getParent(closedlist, currentLocation.SCATS):
                SCAT.append(p.SCATS)
                Location.append(p.Location)
                Latitude.append(p.Latitude)
                Longitude.append(p.Longitude)
            break

        for neighbours in getneighbors(currentLocation.SCATS):
            g = currentLocation.G + getdistance(currentLocation.SCATS, neighbours.SCATS)
            h = getdistance(neighbours.SCATS, endIndex)
            f = g + h
            heapq.heappush(openlist, (f, Node(neighbours.SCATS, f, g, h, currentLocation.SCATS)))

    return SCAT, Location, Latitude, Longitude


def getsecond(o, d):
    startIndex = o
    endIndex = d

    # pass in the available nodes here need to refractor
    Node = collections.namedtuple("Node", "SCATS,F,G,H,parentSCATS")
    h = getdistance(startIndex, endIndex)
    openlist = [(h, Node(startIndex, h, 0, h, None))]  # heap
    closedlist = {}  # map visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    test = ['4040', '3180', '4057', '3682', '2825', '3804', '4034']
    while len(openlist) >= 1:
        _, currentLocation = heapq.heappop(openlist)
        print(currentLocation)

        if currentLocation.SCATS in closedlist:
            continue
        closedlist[currentLocation.SCATS] = currentLocation.parentSCATS
        if currentLocation.SCATS == endIndex:
            print("Complete")
            for p in getParent(closedlist, currentLocation.SCATS):
                SCAT.append(p.SCATS)
                Location.append(p.Location)
                Latitude.append(p.Latitude)
                Longitude.append(p.Longitude)
            break

        for neighbours in getneighbors(currentLocation.SCATS, 9):
            if neighbours.SCATS in test:
                weight = 200
                for next in getnewneighbors(neighbours.SCATS, 2, weight):
                    g = currentLocation.G + getdistance(neighbours.SCATS, next.SCATS)
                    h = getdistance(next.SCATS, endIndex)
                    f = g + h
                    heapq.heappush(openlist, (f, Node(next.SCATS, f, g, h, currentLocation.SCATS)))
            else:
                g = currentLocation.G + getdistance(currentLocation.SCATS, neighbours.SCATS)
                h = getdistance(neighbours.SCATS, endIndex)
                f = g + h
                heapq.heappush(openlist, (f, Node(neighbours.SCATS, f, g, h, currentLocation.SCATS)))

    return SCAT, Location, Latitude, Longitude


'''
def getthird(o, d):
    startIndex = o
    endIndex = d

    # pass in the available nodes here need to refractor
    Node = collections.namedtuple("Node", "SCATS,F,G,H,parentSCATS")
    h = getdistance(startIndex, endIndex)
    openlist = [(h, Node(startIndex, h, 0, h, None))]  # heap
    closedlist = {}  # map visited nodes to parent
    SCAT = []
    Location = []
    Latitude = []
    Longitude = []
    test = ['4040', '3180', '4057', '3682', '2825', '3804', '4034']
    while len(openlist) >= 1:
        _, currentLocation = heapq.heappop(openlist)
        print(currentLocation)

        if currentLocation.SCATS in closedlist:
            continue
        closedlist[currentLocation.SCATS] = currentLocation.parentSCATS
        if currentLocation.SCATS == endIndex:
            print("Complete")
            for p in getParent(closedlist, currentLocation.SCATS):
                SCAT.append(p.SCATS)
                Location.append(p.Location)
                Latitude.append(p.Latitude)
                Longitude.append(p.Longitude)
            break

        for neighbours in getneighbors(currentLocation.SCATS, 9):
            if neighbours.SCATS in test:
                weight = 200
                for next in getnewneighbors(neighbours.SCATS, 2, weight):
                    g = currentLocation.G + getdistance(neighbours.SCATS, next.SCATS)
                    h = getdistance(next.SCATS, endIndex)
                    f = g + h
                    heapq.heappush(openlist, (f, Node(next.SCATS, f, g, h, currentLocation.SCATS)))
            else:
                g = currentLocation.G + getdistance(currentLocation.SCATS, neighbours.SCATS)
                h = getdistance(neighbours.SCATS, endIndex)
                f = g + h
                heapq.heappush(openlist, (f, Node(neighbours.SCATS, f, g, h, currentLocation.SCATS)))

    return SCAT, Location, Latitude, Longitude
    '''

first = getfirst('0970', '4321')
SC, Loc, Lat, Long = first

for i in range(len(Loc)):
    print(SC[i], Loc[i], Lat[i], Long[i])

rawdata = getsecond('0970', '4321')
SC2, Loc2, Lat2, Long2 = rawdata

for i in range(len(Loc2)):
    print(SC2[i], Loc2[i], Lat2[i], Long2[i])
