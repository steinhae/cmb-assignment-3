#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import collections
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

from .util import read_csv, CircleGeofence, get_formated, get_percentage_formated
from .constants import locations, measurements_path, graphs_path, generated_measurements_path
from .geofence_util import get_measurements_per_geofence
from .split_by_geofence import write_geofence_measurements


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', '--report', choices=['stats', '1', 'splitByGeofence', '2',
                                                   'plotDist', '3', 'report', '4', 'vendor/provider plots', '5'],
                        help='available report types\n'
                             '1 - print: statistics\n'
                             '2 - write: measurements for geofences\n'
                             '3 - plot: location distributions\n'
                             '4 - data rate/time report\n'
                             '5 - plot: vendor / provider box- and pie plots',
                        required=True)
    parser.add_argument('-s', '--show-plots', dest='show_plots', action='store_true',
                        help='show the plots instead of just saving them')
    args = parser.parse_args()

    if not os.path.exists(generated_measurements_path):
        os.makedirs(generated_measurements_path)

    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)

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
        elif args.report == '5':
            plot_boxplots_for_locations(measurements, 'all')

            wifi_measurements, mobile_data_measurements = split_wifi_mobile_data(measurements)

            plot_boxplots_for_locations(wifi_measurements, 'wifi')
            plot_boxplots_for_locations(mobile_data_measurements, 'mobile')

            vendors = get_vendors_dict(measurements)

            # vendor plots
            plot_boxplots_for_dict(vendors)
            # vendors pie
            plot_pie_for_dict(vendors)

            providers = get_measurements_per_provider(mobile_data_measurements)

            # provider plots
            plot_boxplots_for_dict(providers)
            # provider pie
            plot_pie_for_dict(providers)


def plot_pie_for_dict(dct):
    names = dct.keys()
    values = [len(v_m) for v_m in dct.values()]
    total_sum = float(sum(values))
    percentages = [value / total_sum for value in values]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    colors = ['#0486e7', '#fadc4a', '#ff0000', '#00ff00']
    ax.pie(percentages, colors=colors[:len(percentages)], labels=names,
           autopct='%1.1f%%', pctdistance=0.6, textprops={'fontsize': 14})
    plt.savefig(os.path.join(graphs_path, 'providers-pie.png'), format='png', dpi=400)
    plt.close(fig)


def plot_boxplots_for_dict(dct):
    for name, measurements in dct.items():
        download, upload, _ = get_downlink_uplink_as_array(measurements)
        plot_boxplot(download, name + '-donwload-{}'.format(len(measurements)))
        plot_boxplot(upload, name + '-upload-{}'.format(len(measurements)))


def split_wifi_mobile_data(measurements):
    wifi_measurements = []
    mobile_data_measurements = []
    for measurement in measurements:
        if measurement['radiotech'] == '0':
            wifi_measurements.append(measurement)
        else:
            mobile_data_measurements.append(measurement)

    return wifi_measurements, mobile_data_measurements


def plot_boxplots_for_locations(measurements, name):
    download, upload, _ = get_downlink_uplink_as_array(measurements)
    plot_boxplot(download, name + '-' + 'download-{}'.format(len(download)))
    plot_boxplot(upload, name + '-' + 'upload-{}'.format(len(upload)))
    for location_key, location in locations.items():
        measurements_for_location = get_measurements_per_geofence(
            CircleGeofence(location, 500),
            measurements)

        plot_hour_comparision(measurements_for_location, name + '-' + location_key)

        download, upload, _ = get_downlink_uplink_as_array(measurements_for_location)
        plot_boxplot(download, name + '-' + location_key + '-download-{}'.format(len(download)))
        plot_boxplot(upload, name + '-' + location_key + '-upload-{}'.format(len(upload)))


def data_rate_statistic_report(measurements):
    for location_key, location in locations.items():
        measurements_for_location = get_measurements_per_geofence(CircleGeofence(location, 500),
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
    for index, measurement in enumerate(measurements_for_location):
        if measurement['downlink'] != '':
            plot_values.append(measurement['downlink'])
            connection_type = 'W' if measurement['radiotech'] == '0' else 'C'
            # labels.append(measurement['startedAt'][10:len(measurement['startedAt'])-3])

    # Reverse the lists/labels so they are in ascending measurement order
    plot_values.reverse()
    labels = [str(i + 1) for i in range(0, len(plot_values))]
    x_labels = ['0']

    if len(labels) > 10:
        tick = 5
        if len(labels) > 100:
            tick = 10
        for index in range(0, len(labels) + tick - 1):
            if (index + 1) % tick == 0:
                if index >= len(labels):
                    x_labels.append(str(index + 1))
                    break
                x_labels.append(labels[index])
            else:
                x_labels.append('')
    else:
        x_labels = labels

    axis_bgcolor = '#f0f0f0'
    figure, ax1 = plt.subplots()

    ax1.plot(plot_values)
    ax1.set_xlim(xmin=0)
    ax1.set_ylim(ymin=0)
    ax1.set_xticks(map(lambda x: x, range(0, len(x_labels) + 1)))
    ax1.set_xticklabels(x_labels, rotation=45, rotation_mode='anchor', ha='right')
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True)

    ax1.set_facecolor(axis_bgcolor)
    ax1.set_title('Downlink / time: ' + name)
    ax1.set_xlabel('Measurement number')
    # ax.axes.get_xaxis().set_visible(False)
    ax1.set_ylabel('Data rate in kbit/s')

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_path, name + '_downlink_time.png'), format='png', dpi=400)
    plt.show()
    plt.close(figure)


def plot_hour_comparision(measurements, name):
    hours_dict = collections.defaultdict(list)
    if not measurements:
        return

    for measurement in measurements:
        time = datetime.datetime.strptime(measurement['startedAt'], '%Y-%m-%d %H:%M:%S')
        hours_dict[time.hour].append(measurement)

    hours_average_list = []

    for hour, hour_measurements in hours_dict.items():
        downlink, uplink, _ = get_downlink_uplink_as_array(hour_measurements)
        hours_average_list.append((hour, (sum(downlink)/float(len(downlink)), sum(uplink)/float(len(uplink)))))

    hours_average_list.sort(key=lambda x: x[0])
    labels, averages = tuple(zip(*hours_average_list))
    labels = list(labels)
    averages = list(averages)

    for i in range(0, 24):
        if i not in labels:
            labels.insert(i, i)
            averages.insert(i, (0,0))


    fig, ax = plt.subplots()

    first_pos = [l*1.4 for l in labels]
    sec_pos = [l + 0.6 for l in first_pos]
    x_ticks = [l + 0.3 for l in first_pos]
    ax.bar(first_pos, [avg[0] for avg in averages], width=0.6, label='downlink')
    ax.bar(sec_pos, [avg[1] for avg in averages], width=0.6, label='uplink')
    ax.set_xticks(x_ticks)
    for i in range(0, len(labels)):
        if labels[i] in hours_dict:
            labels[i] = str(labels[i]) + '\n({})'.format(len(hours_dict[labels[i]]))
    ax.set_xticklabels(labels)
    ax.set_xlim(xmin=-0.5, xmax=max(x_ticks) + 0.8)
    ax.set_ylim(ymin=0)

    ax.set_title('Average up/downlink per hour of day')
    ax.set_xlabel('Hour of day (number of measurements per period in brackets)')
    ax.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(graphs_path, name + '-hourly-up-downlink.png'), format='png', dpi=400)

    plt.close(fig)


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
    plt.savefig(os.path.join(graphs_path, name + '_data_rate_statistic.png'), format='png', dpi=400)


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


def plot_boxplot(data, name):
    if not data:
        print('{} contains no data'.format(name))
        return

    data.sort()
    fig, ax = plt.subplots()
    median = np.median(data)
    lower_quartil = np.percentile(data, 25)
    upper_quartil = np.percentile(data, 75)
    lower_whisker = np.percentile(data, 2.5)
    upper_whisker = np.percentile(data, 97.5)

    ax.boxplot(data, 0, 'gD')
    ax.set_ylabel('Data rate')
    plt.savefig(os.path.join(graphs_path, name + '-boxplot.png'), format='png', dpi=400)
    plt.close(fig)



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

    plt.savefig(graphs_path + 'clustered_measurements_vendor.png', format='png', dpi=400)
    if show_plots:
        plt.show()


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def print_measurement_statistics_report(measurements):
    overall = len(measurements)
    clean = clean_measurements(measurements)
    clean_len = len(clean)
    wifi = len(list(filter(lambda x: x['radiotech'] == '0', clean)))
    print('Statistics:')
    print('# measurements = {}'.format(overall))
    print('# measurements clean = {}'.format(clean_len))
    print('# wifi clean = {} {}'.format(wifi, get_percentage_formated(wifi, clean_len)))
    print('# cellular clean = {} {}'.format(clean_len-wifi, get_percentage_formated(clean_len-wifi, clean_len)))
    print_measurements_per_vendor(measurements)


def get_vendors_dict(measurements):
    vendors = collections.defaultdict(list)
    for measurement in clean_measurements(measurements):
        vendors[measurement['vendor']].append(measurement)

    return vendors


def print_measurements_per_vendor(measurements):
    vendors = get_vendors_dict(measurements)

    total_sum = sum([len(v) for v in vendors.itervalues()])  # excludes measurements with uplink and downlink = 0

    for vendor_name in vendors:
        share = len(vendors[vendor_name])
        print('# {} = {} {}'.format(vendor_name, share,
                                    get_percentage_formated(share, total_sum)))


def get_measurements_per_vendor(vendor, data):
    result = []
    for line in data:
        if line['vendor'] == vendor:
            result.append(line)
    return result


def get_measurements_per_provider(measurements):
    providers = collections.defaultdict(list)

    for measurement in measurements:
        if measurement['network.operator'] and measurement['network.operator'] != '-':
            providers[measurement['network.operator']].append(measurement)

    return providers


def clean_measurements(measurements):
    return [measurement for measurement in measurements
                      if measurement['downlink'] and measurement['uplink'] and
                      float(measurement['downlink']) > 0 and float(measurement['downlink']) > 0]


if __name__ == '__main__':
    main()
