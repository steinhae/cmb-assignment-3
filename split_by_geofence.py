#!/usr/bin/env python3

import os

from .util import write_csv
from .constants import geofences, generated_measurements_path
from .geofence_util import get_measurements_per_geofence


def write_geofence_measurements(measurements, data_fields):
    for geofence in geofences:
        write_csv_for_geofence(measurements, geofence, data_fields)


def write_csv_for_geofence(measurements, geofence, data_fields):
    result = get_measurements_per_geofence(geofence, measurements)
    file_name = os.path.join(generated_measurements_path,
                             geofence.name + '_radius_' + str(geofence.radius) + 'm.csv')
    write_csv(file_name, result, data_fields)
    print('{} measurements have been written to {}'.format(len(result), file_name))
