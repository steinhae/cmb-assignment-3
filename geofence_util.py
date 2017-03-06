#!/usr/bin/env python3

def get_measurements_per_geofence(geofence, data):
    result = []
    for line in data:
        lat = line['latitude']
        lon = line['longitude']
        if lat is not '-' and lon is not '-':
            if geofence.is_location_inside(float(lat), float(lon)):
                result.append(line)
    return result
