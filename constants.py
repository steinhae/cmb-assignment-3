#!/usr/bin/env python3

from .util import Location, CircleGeofence

# Path config
measurements_path = 'cmb-assignment-3/measurements/'
generated_measurements_path = 'cmb-assignment-3/measurements/generated/'
graphs_path = 'cmb-assignment-3/graphs/'

# Location config
locations = {
    'tum_mi': Location('tum_mi', 48.262547, 11.667838),
    'hannes_home': Location('hannes_home', 51.806646, 10.456883),
    'patricks_home': Location('patricks_home', 46.344574, 11.241863),
    'munich_area': Location('munich_area', 48.138561, 11.573757)
}

geofences = {
    'munich_area': CircleGeofence(locations['munich_area'], 7000),
    'tum_mi': CircleGeofence(locations['tum_mi'], 150),
    'hannes_home': CircleGeofence(locations['hannes_home'], 35000),
    'patricks_home': CircleGeofence(locations['patricks_home'], 35000)
}
