#!/usr/bin/env python

import csv
import argparse
import numpy as np
import pandas as pd, matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from geopy.distance import vincenty
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

axis_bgcolor = '#f0f0f0'
title_font = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=15, weight='normal',
                               stretch='normal')
label_font = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=12, weight='normal',
                               stretch='normal')
ticks_font = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=10, weight='normal',
                               stretch='normal')
annotation_font = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=10, weight='normal',
                                    stretch='normal')

class CircleGeofence():
    def __init__(self, name, latitude, longitude, radius):
        self.longitude = longitude
        self.latitude = latitude
        self.radius = radius
        self.name = name

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', '--report',
                        help='available report types\n'
                            '1 - statistics\n'
                            '2 - write measurements for geofences\n'
                            '3 - plot location distributions',
                        required=True)
    args = parser.parse_args()

    measurements, data_fields = read_csv('measurements_clean.csv')

    rn = (vars(args))['report']
    if rn == '1':
        print_measurement_statistics_report(measurements)
    elif rn == '2':
        write_geofence_measurements(measurements, data_fields)
    elif rn == '3':
        plot(measurements)

def plot(measurements):
    # Credits to Geoff Boeing
    # @http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
    df = pd.read_csv('measurements_clean.csv')
    df['latitude'].replace('-', np.nan, inplace=True)
    df.dropna(subset=['latitude'], inplace=True)
    df['longitude'].replace('-', np.nan, inplace=True)
    df.dropna(subset=['longitude'], inplace=True)
    df.convert_objects(convert_numeric=True)
    df.longitude = df.longitude.astype(float).fillna(0.0)
    df.latitude = df.latitude.astype(float).fillna(0.0)
    coords = df.as_matrix(columns=['latitude', 'longitude'])

    kms_per_radian = 6371.0088
    epsilon = 2.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    print('Number of clusters: {}'.format(num_clusters))

    centermost_points = clusters.map(get_centermost_point)
    lats, lons = zip(*centermost_points)
    rep_points = pd.DataFrame({'longitude': lons, 'latitude': lats})
    rs = rep_points.apply(lambda row: df[(df['latitude'] == row['latitude']) & (df['longitude'] == row['longitude'])].iloc[0], axis = 1)

    fig, ax = plt.subplots(figsize=[10, 6])
    rs_scatter = ax.scatter(rs['longitude'], rs['latitude'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
    df_scatter = ax.scatter(df['longitude'], df['latitude'], c='k', alpha=0.9, s=3)
    ax.set_title('Full data set vs DBSCAN clustered set')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend([df_scatter, rs_scatter], ['Full set', 'Clustered set'], loc='upper right')

    # #annotate clusters with vendor names
    for i, row in rs.iterrows():
        ax.annotate(row['vendor'],
                xy=(row['longitude'], row['latitude']),
                xytext=(row['longitude']+0.05, row['latitude']+0.15),
                fontproperties=annotation_font,
                bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.8),
                xycoords='data',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='k', alpha=0.8))

    plt.savefig('clustered_measurements_vendor.png')
    plt.show()

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def write_geofence_measurements(measurements, data_fields):
    geofences = []
    geofences.append(CircleGeofence('munich_area', 48.138561, 11.573757, 7000))
    geofences.append(CircleGeofence('tum_mi', 48.262547, 11.667838, 150))
    for geofence in geofences:
        write_csv_for_geofence(measurements, geofence, data_fields)

def print_measurement_statistics_report(measurements):
    overall = len(measurements)
    wifi = 0
    for line in measurements:
        if line['radiotech'] == '0':
            wifi = wifi + 1
    print 'Statistics:'
    print '# measurements =', overall
    print '# wifi = ', wifi, get_percentage_formated(wifi, overall)
    print '# cellular =', overall-wifi, get_percentage_formated(wifi, overall)
    print_measurements_per_vendor(measurements)

def write_csv_for_geofence(measurements, geofence, data_fields):
    result = get_measurements_per_geofence(geofence, measurements)
    file_name = geofence.name + '_radius_' + str(geofence.radius) + 'm.csv'
    write_csv(file_name, result, data_fields)
    print len(result), 'measurements have been written to', file_name

def print_measurements_per_vendor(measurements):
    vendors = []
    for line in measurements:
        if line['vendor'] not in vendors:
            vendors.append(line['vendor'])
    for vendor in vendors:
        share = len(get_measurements_per_vendor(vendor, measurements))
        print '#', vendor, '=', share, get_percentage_formated(share, len(measurements))


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

def get_percentage_formated(share, amount):
    return '(' + "%.2f" % (float(share) / float(amount) * 100) + '%)'

if __name__ == '__main__':
    main()