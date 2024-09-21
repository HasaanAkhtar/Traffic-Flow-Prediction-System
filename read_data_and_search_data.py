import csv
from collections import defaultdict

from flask import Flask

app = Flask(__name__, template_folder='template')
scat = []
columns = defaultdict(list)
data = {}
location = []


# read entire csv file
def readfile():
    with open('template/SCAT.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            for (i, v) in enumerate(row):
                columns[i].append(v)
    for x in (columns[0]):
        scat.append(x)


# search for scats and retrieve corresponding coordinates
def search(search):
    readfile()
    return searchstring(search)


# search coordinates based on scat passed
def getcoordinates(s):
    latitude = []
    longitude = []
    for l in columns[2]:
        latitude.append(l)
    for g in columns[3]:
        longitude.append(g)
    #    print(latitude[s], longitude[s])
    coordinates = (latitude[s], longitude[s])
    return coordinates


# search for intersection name to check if it exists
def searchloc(search):
    readfile()
    found = False
    for lo in columns[1]:
        location.append(lo)
    for s in range(len(location) - 1):
        if search in location[s]:
            found = True
    return found


# search for scat to check if it exists
def searchScat(search):
    found = False
    readfile()
    for s in range(len(scat) - 1):
        if search in scat[s]:
            found = True
    return found


def searchstring(search):
    # print('Im here')
    for s in range(len(scat)):
        if search in scat[s]:
            dataset = getcoordinates(s)
            return dataset

#print(searchScat('0000'))
#print(searchloc('HIGH STREET_RD E of WARRIGAL_RD'))
