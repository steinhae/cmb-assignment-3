#!/usr/bin/env python

import csv
import argparse
import numpy as np
import pandas as pd, matplotlib.pyplot as plt, matplotlib.font_manager as fm
from geopy.distance import vincenty
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

class Location():
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

class CircleGeofence():
    def __init__(self, location, radius):
        self.longitude = location.longitude
        self.latitude = location.latitude
        self.radius = radius
        self.name = location.name

# Path config
measurements_path = 'measurements/'
generated_measurements_path = 'measurements/generated/'
graphs_path = 'graphs/'

# Location config
location = {}
location['tum_mi'] = Location('tum_mi', 48.262547, 11.667838)
location['hannes_home'] = Location('hannes_home', 51.806646, 10.456883)
location['patricks_home'] = Location('patricks_home', 46.344574, 11.241863)
location['munich_area'] = Location('munich_area', 48.138561, 11.573757)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', '--report',
                        help='available report types\n'
                            '1 - statistics\n'
                            '2 - write measurements for geofences\n'
                            '3 - plot location distributions\n'
                            '4 - data rate/time report',
                        required=True)
    args = parser.parse_args()

    measurements, data_fields = read_csv(measurements_path + 'measurements_clean.csv')

    rn = (vars(args))['report']
    if rn == '1':
        print_measurement_statistics_report(measurements)
    elif rn == '2':
        write_geofence_measurements(measurements, data_fields)
    elif rn == '3':
        plot_clustered_measurements_vendor(measurements)
    elif rn == '4':
        data_rate_statistic_report(measurements)

def data_rate_statistic_report(measurements):
    for l in location:
        measurements_for_location = get_measurements_per_geofence(CircleGeofence(location[l], 500), measurements)
        if len(measurements_for_location) > 0:

            downlink, uplink, measurements_used = get_downlink_uplink_as_array(measurements_for_location)
            name = location[l].name

            print 'Data rate statistic for', name, '(' + str(measurements_used) + '/' + str(
                len(measurements_for_location)), 'measurements used, uplink and downlink != 0):'

            print '[WI-FI + Cellular]'
            print_plot_data_rate_statistic(name, downlink, uplink, len(measurements_for_location))

            plot_data_rate_time(name, measurements_for_location)

def plot_data_rate_time(name, measurements_for_location):

    # TODO: seperate WIFI and Cellular (one data layer for each)
    plot_values = []
    labels = []
    for measurement in measurements_for_location:
        if measurement['downlink'] != '':
            plot_values.append(measurement['downlink'])
            #labels.append(measurement['startedAt'][10:len(measurement['startedAt'])-3])
            labels.append(measurement['startedAt'])

    # Reverse the lists/labels so they are in ascending measurement order
    plot_values.reverse()
    labels.reverse()
    axis_bgcolor = '#f0f0f0'
    figure, ax = plt.subplots()

    ax.plot(plot_values)
    ax.set_xticks(map(lambda x: x, range(0, len(plot_values))))
    ax.set_xticklabels(labels, rotation=45, rotation_mode='anchor', ha='right')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    ax.set_facecolor(axis_bgcolor)
    ax.set_title('Data rate time: ' + name)
    ax.set_xlabel('Time')
    #ax.axes.get_xaxis().set_visible(False)
    ax.set_ylabel('Data rate in kbit/s')
    plt.tight_layout()
    plt.savefig(graphs_path + name + '_data_rate_time.png')
    plt.show()

def print_plot_data_rate_statistic(name, downlink, uplink, count):

    unit = 'kbit/s'
    data = []

    average_downlink = sum(downlink) / count
    print 'Average downlink = ', get_formated(average_downlink), unit
    data.append((average_downlink, 'Average downlink'))

    average_uplink = sum(uplink) / count
    print 'Average uplink   = ', get_formated(average_uplink), unit
    data.append((average_uplink, 'Average uplink'))

    print 'Maximum downlink = ', get_formated(max(downlink)), unit
    data.append((max(downlink), 'Maximum downlink'))

    print 'Maximum uplink   = ', get_formated(max(uplink)), unit
    data.append((max(uplink), 'Maximum uplink'))

    print 'Minimum downlink = ', get_formated(min(downlink)), unit
    data.append((min(downlink), 'Minimum downlink'))

    print 'Minimum uplink   = ', get_formated(min(uplink)), unit
    data.append((min(uplink), 'Minimum uplink'))

    print ''

    data_rate_statistic_plot(name, data)

def data_rate_statistic_plot(name, data):
    plot_values = [x[0] for x in data]
    labels =  [x[1] for x in data]

    ind = np.arange(len(data))

    axis_bgcolor = '#f0f0f0'
    figure, ax = plt.subplots()
    #ax.plot(plot_values)
    ax.bar(ind, plot_values, 0.35)
    ax.set_xticks(map(lambda x: x, range(0, len(data))))
    ax.set_xticklabels(labels, rotation=45, rotation_mode='anchor', ha='right')
    ax.yaxis.grid(True)

    ax.set_facecolor(axis_bgcolor)
    ax.set_title('Data rate statistic: ' + name)
    ax.set_xlabel('')
    ax.set_ylabel('Data rate in kbit/s')

    plt.tight_layout()
    plt.savefig(graphs_path + name + '_data_rate_statistic.png')

def get_downlink_uplink_as_array(measurements):
    downlink = []
    uplink = []
    measurements_used = 0
    for measurement in measurements:
        try:
            dl = float(measurement['downlink'])
            up = float(measurement['uplink'])
            if dl != 0 and up != 0:
                downlink.append(dl)
                uplink.append(up)
                measurements_used = measurements_used + 1
        except:
            pass
    return downlink, uplink, measurements_used

def plot_clustered_measurements_vendor(measurements):
    # Credits to Geoff Boeing
    # @http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
    df = pd.read_csv(measurements_path + 'measurements_clean.csv')
    df['latitude'].replace('-', np.nan, inplace=True)
    df.dropna(subset=['latitude'], inplace=True)
    df['longitude'].replace('-', np.nan, inplace=True)
    df.dropna(subset=['longitude'], inplace=True)
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
                bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.8),
                xycoords='data',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='k', alpha=0.8))

    plt.savefig(graphs_path + 'clustered_measurements_vendor.png')
    plt.show()

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

def write_geofence_measurements(measurements, data_fields):
    geofences = []
    geofences.append(CircleGeofence(location['munich_area'], 7000))
    geofences.append(CircleGeofence(location['tum_mi'], 150))
    geofences.append(CircleGeofence(location['hannes_home'], 35000))
    geofences.append(CircleGeofence(location['patricks_home'], 35000))
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
    file_name = generated_measurements_path + geofence.name + '_radius_' + str(geofence.radius) + 'm.csv'
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

def get_formated(value):
    return "%.2f" % value

def get_percentage_formated(share, amount):
    return '(' + "%.2f" % (float(share) / float(amount) * 100) + '%)'

if __name__ == '__main__':
    main()