#!/usr/bin/env python3

from geopy.distance import vincenty
import csv


class CircleGeofence:
    def __init__(self, location, radius):
        self.longitude = location.longitude
        self.latitude = location.latitude
        self.radius = radius
        self.name = location.name

    def is_location_inside(self, lat, lon):
        return vincenty((lat, lon), (self.latitude, self.longitude)).meters < self.radius


class Location:
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

    def is_inside_geofence(self, geofence):
        return vincenty((self.latitude, self.longitude),
                        (geofence.latitude, geofence.longitude)).meters < geofence.radius


def write_csv(file_name, measurements, data_fields):
    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_fields)
        writer.writeheader()
        writer.writerows(measurements)


def read_csv(file_name):
    with open(file_name, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        fields = reader.fieldnames
        list_out = [row for row in reader]
        return list_out, fields


def get_formated(value):
    return "%.2f" % value


def get_percentage_formated(share, amount):
    return '(' + "%.2f" % (float(share) / float(amount) * 100) + '%)'
