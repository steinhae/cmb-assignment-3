#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

from .util import read_csv, CircleGeofence, get_formated, get_percentage_formated
from .constants import locations, measurements_path, graphs_path
from .geofence_util import get_measurements_per_geofence
from .split_by_geofence import write_geofence_measurements


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', '--report', choices=['stats', '1', 'splitByGeofence', '2',
                                                   'plotDist', '3', 'report', '4'],
                        help='available report types\n'
                             '1 - statistics\n'
                             '2 - write measurements for geofences\n'
                             '3 - plot location distributions\n'
                             '4 - data rate/time report',
                        required=True)
    parser.add_argument('-s', '--show-plots', dest='show_plots', action='store_true',
                        help='show the plots instead of just saving them')
    args = parser.parse_args()

    if args.report == '3':
        plot_clustered_measurements_vendor(args.show_plots)
    else:
        measurements, data_fields = read_csv(
            os.path.join(measurements_path, 'measurements_clean.csv'))
        if args.report == '1':
            print_measurement_statistics_report(measurements)
        elif args.report == '2':
            write_geofence_measurements(measurements, data_fields)
        elif args.report == '4':
            data_rate_statistic_report(measurements)


def data_rate_statistic_report(measurements):
    for location_key, location in locations.items():
        measurements_for_location = get_measurements_per_geofence(CircleGeofence(location_key, 500),
                                                                  measurements)
        if len(measurements_for_location) > 0:

            downlink, uplink, measurements_used = get_downlink_uplink_as_array(
                measurements_for_location)
            name = location.name

            print('Data rate statistic for {} ( {} / {} measurements used, '
                  'uplink and downlink != 0):'.format(name, measurements_used,
                                                      len(measurements_for_location)))

            print('[WI-FI + Cellular]')
            print_plot_data_rate_statistic(name, downlink, uplink, len(measurements_for_location))

            plot_data_rate_time(name, measurements_for_location)


def plot_data_rate_time(name, measurements_for_location):
    plot_values = []
    labels = []
    for measurement in measurements_for_location:
        if measurement['downlink'] != '':
            plot_values.append(measurement['downlink'])
            connection_type = 'W' if measurement['radiotech'] == '0' else 'C'
            # labels.append(measurement['startedAt'][10:len(measurement['startedAt'])-3])
            labels.append(measurement['startedAt'] + ' (' + connection_type + ')')

    # Reverse the lists/labels so they are in ascending measurement order
    plot_values.reverse()
    labels.reverse()
    axis_bgcolor = '#f0f0f0'
    figure, ax1 = plt.subplots()

    ax1.plot(plot_values)
    ax1.set_xticks(map(lambda x: x, range(0, len(plot_values))))
    ax1.set_xticklabels(labels, rotation=45, rotation_mode='anchor', ha='right')
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True)

    ax1.set_facecolor(axis_bgcolor)
    ax1.set_title('Downlink / time: ' + name)
    ax1.set_xlabel('Time')
    # ax.axes.get_xaxis().set_visible(False)
    ax1.set_ylabel('Data rate in kbit/s')

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_path, name + '_downlink_time.pdf'), format='pdf', dpi=2000)
    plt.show()


def print_plot_data_rate_statistic(name, downlink, uplink, count):

    unit = 'kbit/s'
    data = []

    average_downlink = sum(downlink) / count
    print('Average downlink = {} {}'.format(get_formated(average_downlink), unit))
    data.append((average_downlink, 'Average downlink'))

    average_uplink = sum(uplink) / count
    print('Average uplink   = {} {}'.format(get_formated(average_uplink), unit))
    data.append((average_uplink, 'Average uplink'))

    print('Maximum downlink = {} {}'.format(get_formated(max(downlink)), unit))
    data.append((max(downlink), 'Maximum downlink'))

    print('Maximum uplink   = {} {}'.format(get_formated(max(uplink)), unit))
    data.append((max(uplink), 'Maximum uplink'))

    print('Minimum downlink = {} {}'.format(get_formated(min(downlink)), unit))
    data.append((min(downlink), 'Minimum downlink'))

    print('Minimum uplink   = {} {}'.format(get_formated(min(uplink)), unit))
    data.append((min(uplink), 'Minimum uplink'))

    print('')

    data_rate_statistic_plot(name, data)


def data_rate_statistic_plot(name, data):
    plot_values = [x[0] for x in data]
    labels = [x[1] for x in data]

    ind = np.arange(len(data))

    axis_bgcolor = '#f0f0f0'
    figure, ax = plt.subplots()
    # ax.plot(plot_values)
    ax.bar(ind, plot_values, 0.35)
    ax.set_xticks(map(lambda x: x, range(0, len(data))))
    ax.set_xticklabels(labels, rotation=45, rotation_mode='anchor', ha='right')
    ax.yaxis.grid(True)

    ax.set_facecolor(axis_bgcolor)
    ax.set_title('Data rate statistic: ' + name)
    ax.set_xlabel('')
    ax.set_ylabel('Data rate in kbit/s')

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_path, name + '_data_rate_statistic.pdf'), format='pdf', dpi=2000)


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
                measurements_used += 1
        except:
            pass
    return downlink, uplink, measurements_used


def plot_clustered_measurements_vendor(show_plots):
    # Credits to Geoff Boeing
    # @http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
    df = pd.read_csv(os.path.join(measurements_path, 'measurements_clean.csv'))
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

    plt.savefig(graphs_path + 'clustered_measurements_vendor.pdf', format='pdf', dpi=2000)
    if show_plots:
        plt.show()


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def print_measurement_statistics_report(measurements):
    overall = len(measurements)
    wifi = 0
    for line in measurements:
        if line['radiotech'] == '0':
            wifi = wifi + 1
    print('Statistics:')
    print('# measurements = {}'.format(overall))
    print('# wifi = {} {}'.format(wifi, get_percentage_formated(wifi, overall)))
    print('# cellular = {} {}'.format(overall-wifi, get_percentage_formated(wifi, overall)))
    print_measurements_per_vendor(measurements)


def print_measurements_per_vendor(measurements):
    vendors = []
    for line in measurements:
        if line['vendor'] not in vendors:
            vendors.append(line['vendor'])
    for vendor in vendors:
        share = len(get_measurements_per_vendor(vendor, measurements))
        print('# {} = {} {}'.format(vendor, share, get_percentage_formated(share, len(measurements))))


def get_measurements_per_vendor(vendor, data):
    result = []
    for line in data:
        if line['vendor'] == vendor:
            result.append(line)
    return result


if __name__ == '__main__':
    main()
