from dataclasses import dataclass
import geopy.distance
import folium
from flask import render_template, request, url_for, Flask
from werkzeug.utils import redirect
from read_data_and_search_data import *
from Scat_pathfinding import *
from intersection_name_pathfinding import *

app = Flask(__name__, template_folder='template')


# dataclass for data that can be accessible throughout the program
@dataclass
class DataClass:
    # scat numbers for origin and destination
    scat_origin: str
    scat_dest: str
    # scat location name for orign and destination
    origin_name: str
    destination_name: str


# to render template from HTML form
@app.route("/")
def home():
    return render_template('index.html')


# getting the data using API calls
@app.route("/index", methods=['GET', 'POST'])
def index():
    DataClass.scat_origin = request.args.get('scat_origin', type=str)
    DataClass.scat_dest = request.args.get('scat_dest', type=str)

    DataClass.origin_name = request.args.get('o_name', type=str)
    DataClass.destination_name = request.args.get('d_name', type=str)

    return redirect(url_for('map_marker'))

# to plot the map and markers.
@app.route("/map_marker")
def map_marker():
    # was suppose to get data from machine learning here in this list.
    avoid_scats = ['4040', '3180', '4057', '3682', '2825', '3804', '4034', '3662', '3812']
    '''avoid_loc = ['BURKE_RD N of CANTERBURY_RD', 'BALWYN_RD S of DONCASTER_RD', 'KILBY_RD W of BURKE_RD',
                 'BALWYN_RD N OF BELMORE_RD', 'WARRIGAL_RD S OF RIVERSDALE_RD', 'BURKE_RD S of EASTERN_FWY',
                 'TRAFALGAR_RD S of RIVERSDALE_RD', 'WHITEHORSE_RD E OF BURKE_RD']'''

    if DataClass.origin_name != "":
        origin = searchloc(DataClass.origin_name)
        destination = searchloc(DataClass.destination_name)
        if origin and destination != False:
            first = get_loc(str(DataClass.origin_name), str(DataClass.destination_name))
            second = get_new_loc(str(DataClass.origin_name), str(DataClass.destination_name))
            third = get_third_loc(str(DataClass.origin_name), str(DataClass.destination_name))
            fourth = get_fourth_loc(str(DataClass.origin_name), str(DataClass.destination_name))
        else:
            return redirect(url_for('home'))

    else:
        origin = searchScat(DataClass.scat_origin)
        destination = searchScat(DataClass.scat_dest)
        if origin and destination != False:
            first = get_first(str(DataClass.scat_origin), str(DataClass.scat_dest))
            second = get_second(str(DataClass.scat_origin), str(DataClass.scat_dest))
            third = get_third(str(DataClass.scat_origin), str(DataClass.scat_dest), avoid_scats)
        else:
            return redirect(url_for('home'))

    SCAT, Location, Latitude, Longitude = first

    SCAT2, Location2, Latitude2, Longitude2 = second

    SCAT3, Location3, Latitude3, Longitude3 = third

    SCAT4, Location4, Latitude4, Longitude4 = fourth

    # getting end coordinates index
    k = len(Location) - 1

    # first route
    route = []
    for j in range(len(Location)):
        route.append([Latitude[j], Longitude[j]])
    # print(route)

    # second route
    secondroute = []
    for q in range(len(Location2)):
        secondroute.append([Latitude2[q], Longitude2[q]])
    # print(secondroute)

    # third route
    thirdroute = []
    for q in range(len(Location3)):
        thirdroute.append([Latitude3[q], Longitude3[q]])
    # print(thirdroute)

    fourthroute = []
    for q in range(len(Location4)):
        thirdroute.append([Latitude4[q], Longitude4[q]])
    # print(fourthroute)

    # calculating distance and time with speed set as 60km/hr and getting the time in minutes
    distance_to_destination = geopy.distance.geodesic(route[0], route[k]).km
    distance_to_destination = round(distance_to_destination, 2)
    time = round(((distance_to_destination / 60) * 60), 2)

    # plotting map based on random coordinates
    map = folium.Map(
        location=[-37.850386, 145.095612],
        tiles='Stamen Terrain',
        zoom_start=13
    )
    # adding marker position the origin and the destination
    folium.Marker(location=[Latitude[0], Longitude[0]],
                  popup=Location[0],
                  tooltip='Click here',
                  icon=folium.Icon(color='blue')).add_to(map)
    folium.Marker(location=[Latitude[k], Longitude[k]],
                  popup=Location[k],
                  tooltip='Click here',
                  icon=folium.Icon(color='red')).add_to(map)
    # connecting lines from origin and destination with the routes
    f1 = folium.FeatureGroup("Vehicle 1 path")
    # Adding lines to the different feature groups
    line_1 = folium.vector_layers.PolyLine(route, popup='<b>route</b>', tooltip='<b>distance to destination:' + str(
        distance_to_destination) + 'km</b>'
                                   '<b> time: ' + str(time) + 'minutes</b>',
                                           icon=folium.Icon(color='red'),
                                           color='blue', weight=2).add_to(f1)
    f1.add_to(map)
    f2 = folium.FeatureGroup("Vehicle 2 path")
    # Adding lines to the different feature groups
    line_2 = folium.vector_layers.PolyLine(secondroute, popup='<b>time delay due to traffic</b>', tooltip='route',
                                           icon=folium.Icon(color='red'),
                                           color='blue', weight=2).add_to(f2)
    f2.add_to(map)
    f3 = folium.FeatureGroup("Vehicle 3 path")
    # Adding lines to the different feature groups
    line_3 = folium.vector_layers.PolyLine(thirdroute, popup='<b>route</b>', tooltip='route',
                                           icon=folium.Icon(color='red'),
                                           color='blue', weight=2).add_to(f3)
    f3.add_to(map)

    f4 = folium.FeatureGroup("Vehicle 4 path")
    # Adding lines to the different feature groups
    line_3 = folium.vector_layers.PolyLine(fourthroute, popup='<b>route</b>', tooltip='route',
                                           icon=folium.Icon(color='red'),
                                           color='blue', weight=2).add_to(f4)
    f4.add_to(map)

    return map._repr_html_()


if __name__ == "__main__":
    app.run(debug=True)
