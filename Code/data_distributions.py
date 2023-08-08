"""Defines the data distribution functions"""

import matplotlib.pyplot as plt
from collections import Counter


def description(data):
    # Gets data statistics
    print(data.describe())
    print(data.info())


def data_plot(data, features):
    # Plots the data distribution
    if features == 'age_group':
        values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        data[features].value_counts().loc[values].plot(kind='bar')
    elif features == 'size':
        print(Counter(data['size']))
    else:
        df = data[features]
        keys = df.value_counts().index.to_list()
        keys = [str(i) for i in keys]
        values = df.value_counts().values
        plt.figure(figsize=(18, 12))
        plt.bar(keys, values)
        plt.xticks(keys, rotation='vertical')


def get_data_details(data):
    # Displays the data statistics and plots
    description(data)
    data_plot(data, 'age_group')
    data_plot(data, 'size')
    data_plot(data, ['gender', 'race'])
    data_plot(data, ['gender', 'age_group'])
    data_plot(data, ['race', 'age_group'])
    data_plot(data, ['gender', 'race', 'age_group'])
