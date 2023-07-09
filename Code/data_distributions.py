import matplotlib.pyplot as plt
from collections import Counter

# Get data statistics
def description(data):
    print(data.describe())
    print(data.info())

# Plot the data distribution
def data_plot(data, features):
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

# Main function to call the description and plot functions
def getDataDetails(data):
    description(data)
    #data_plot(data, 'age_group')
    data_plot(data, 'size')
    #data_plot(data, ['gender', 'race'])
    #data_plot(data, ['gender', 'age_group'])
    #data_plot(data, ['race', 'age_group'])
    #data_plot(data, ['gender', 'race', 'age_group'])
