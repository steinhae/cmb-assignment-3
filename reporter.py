#!/usr/bin/env python

import csv
from geopy.distance import vincenty

class CircleGeofence():
    def __init__(self, name, latitude, longitude, radius):
        self.longitude = longitude
        self.latitude = latitude
        self.radius = radius
        self.name = name

def main():
    measurements, data_fields = read_csv('measurements_clean.csv')
    print_number_of_measurements(measurements)
    write_geofence_measurements(measurements, data_fields)

def write_geofence_measurements(measurements, data_fields):
    print_seperator()
    geofences = []
    geofences.append(CircleGeofence('munich_area', 48.138561, 11.573757, 7000))
    geofences.append(CircleGeofence('tum_mi', 48.262547, 11.667838, 150))
    for geofence in geofences:
        write_csv_for_geofence(measurements, geofence, data_fields)

def print_number_of_measurements(measurements):
    overall = len(measurements)
    wifi = 0
    for line in measurements:
        if line['radiotech'] == '0':
            wifi = wifi + 1
    print_seperator()
    print '# measurements =', overall
    print '# wifi = ', wifi, '(' + "%.2f" % (float(wifi)/float(overall)) + '%)'
    print '# cellular =', overall-wifi, '(' + "%.2f" % (float(overall-wifi)/float(overall)) + '%)'


def write_csv_for_geofence(measurements, geofence, data_fields):
    result = get_measurements_per_geofence(geofence, measurements)
    file_name = geofence.name + '_radius_' + str(geofence.radius) + 'm.csv'
    write_csv(file_name, result, data_fields)
    print len(result), 'measurements have been written to', file_name

def print_reports(measurements):
    report_measurements_per_vendor(measurements)

def report_measurements_per_vendor(measurements):
    vendors = []
    for line in measurements:
        if line['vendor'] not in vendors:
            vendors.append(line['vendor'])
    print "Measurements per vendor:"
    for vendor in vendors:
        print vendor, len(get_measurements_per_vendor(vendor, measurements))


def get_measurements_per_geofence(geofence, data):
    result = []
    for line in data:
        lat = line['latitude']
        lon = line['longitude']
        if lat is not '-' and lon is not '-':
            if is_inside(float(lat), float(lon), geofence):
                result.append(line)
    return result

def is_inside(latitude, longitude, geofence):
    if vincenty((latitude, longitude), (geofence.latitude, geofence.longitude)).meters < geofence.radius:
        return True
    else:
        return False

def get_measurements_per_vendor(vendor, data):
    result = []
    for line in data:
        if line['vendor'] == vendor:
            result.append(line)
    return result

def read_csv(file_name):
    with open(file_name, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        fields = reader.fieldnames
        list_out = [row for row in reader]
        return list_out, fields

def write_csv(file_name, measurements, data_fields):
    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_fields)
        writer.writeheader()
        writer.writerows(measurements)

def print_seperator():
    print '---------------------------------------'

if __name__ == '__main__':
    main()
    print_seperator()