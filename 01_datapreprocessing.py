import datetime
import glob
import math
import numpy as np
from numpy import NAN, average, log, roll, rollaxis
import os
import pandas as pd
import matplotlib.pyplot as plt

### PRODUCTION DATA IMPORT ### ----------------------------------------------------

# get all data file names
path = r'data/'  
all_files = glob.glob(path + "/*.csv")
all_files.remove('data/forecasts.csv')

# read all files names and append to a list
temp_list = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, delimiter=';')
    temp_list.append(df)

# concatenate all files to a long 
df_long = pd.concat(temp_list, axis=0, ignore_index=True)
df_long['datetime'] = pd.to_datetime(df_long['datetime'], format='%Y%m%d%H%M%S')
# reshape to wide
prod_data = df_long.pivot(index='datetime', columns='mill_id', values='production')
prod_data.dropna(inplace=True)

# create folder if not yet created
if not os.path.exists('data_processed'):
    os.makedirs('data_processed')

# save to csv
prod_data.to_csv('data_processed/production.csv')


### FORECAST DATA IMPORT ### ----------------------------------------------------

# Import forecasts
forc_long = pd.read_csv('data/forecasts.csv', delimiter=';')
forc_long['datetime'] = pd.to_datetime(forc_long['datetime'], format='%Y%m%d%H%M%S')
forc_wide = forc_long.pivot(index='datetime', values='windspeed_forecast', columns='location_id')

# shift forecast to have prediction of same row
forc_wide = forc_wide.shift(periods=-4)
forc_wide.rename(columns={1:'forc_1_ms', 2:'forc_2_ms'}, inplace=True)
forc_wide.dropna(inplace=True)

# save to csv
forc_wide.to_csv('data_processed/forecasts.csv')


### DATASET A: RAW DATA ### ----------------------------------------------------

# merge df
dataset_a = pd.merge(forc_wide, prod_data, left_on='datetime', right_on='datetime', how='left')

# add 4h in advance production sum as dependent variable
cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
product_sum = dataset_a[cols].sum(axis=1)
dataset_a['prod_sum'] = product_sum.shift(-4)

# drop na forecasts
dataset_a.dropna(inplace=True)

# save to csv
dataset_a.to_csv('data_processed/dataset_a.csv')


### DATASET B: CLEANED POWER ### ----------------------------------------------------

# find days where we have little wind and zero to negative production
a_days = dataset_a.resample(rule='d').mean()
neg_dates = a_days[cols][a_days[cols] < 0].dropna(how='all').index.date

# optional: change output of pandas dataframe
#pd.set_option("display.max_rows", 30, "display.max_columns", None)

# get forecast in m/s for negative dates and plot to find pattern
forc_neg_avg = a_days.loc[neg_dates][['forc_1_ms', 'forc_2_ms']].mean(axis=1)
prod_neg_avg = a_days.loc[neg_dates][cols].mean(axis=1)

forc_neg = a_days.loc[neg_dates][['forc_1_ms', 'forc_2_ms']]
prod_neg = a_days.loc[neg_dates][cols]

# pairplot to see values
a_days_neg = a_days.loc[neg_dates]
# sns.pairplot(a_days_neg)

def plot_scatters(wind, production, nrows=20, ncols=2):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20, 60))
    for i in range(0, len(production.columns)):
        ax[i, 0].scatter(wind['forc_1_ms'], production.iloc[:,i])
        ax[i, 0].set_title('Forecast 1 ' + 'Prod ' + str(production.columns[i]))
        ax[i, 1].scatter(wind['forc_2_ms'], production.iloc[:,i])
        ax[i, 1].set_title('Forecast 2 ' + 'Prod ' + str(production.columns[i]))
    for a in ax.flat:
        a.set(xlabel='m/s', ylabel='MW')
    plt.tight_layout()


plot_scatters(wind=a_days[['forc_1_ms', 'forc_2_ms']], production=a_days[cols])

# we assume that when m/s is above 7.5 and we have negative daily production, that it is not caused by maintenance
# else it may be caused by maintenance or our calculation error
# hence we remove vaues above 7.5 m/s and negative daily production
neg_days_maint = a_days_neg[(a_days_neg['forc_1_ms'] + a_days_neg['forc_1_ms'] / 2) > 7.5]
neg_days_maint = neg_days_maint.index.date

# delete not needed days
dataset_b = dataset_a.copy()
for date in neg_days_maint:
    dataset_b = dataset_b[dataset_b.index.date != date]

print(f"We have deleted in total {len(dataset_a) - len(dataset_b)} rows ({round((len(dataset_a)-len(dataset_b))/len(dataset_a)*100, 1)}%) from Dataset A")

# save dataset c
dataset_b.to_csv('data_processed/dataset_b.csv')


### DATASET C: ADDED TRENDS AND SEASONALITY ### ----------------------------------------------------

dataset_c = dataset_b.copy()
dataset_c.reset_index(inplace=True)
dataset_c['month'] = dataset_c['datetime'].dt.month

# as we have data from january 2015 - april 2017 we can include averages for seasonality
# however we need to avoid forward-looking bias when calculating the average, min, and max
# hence we calculate the averages fixed for the first two years and then we do a rolling average, min, and max

def add_features(df):
    """Adds various features: 
    * Monthly Avg
    * Monthly Min
    * Monthly Max
    """
    # average hour per power per month, average
    month_avg = df.groupby(by=['month']).mean()
    month_avg.drop(columns=['forc_1_ms', 'forc_2_ms'], inplace=True)
    month_avg = month_avg.add_suffix('_avg')

    # min per hour
    month_min = df.groupby(by=['month']).min()
    month_min.drop(columns=['forc_1_ms', 'forc_2_ms'], inplace=True)
    month_min = month_min.add_suffix('_min')

    # max per hour
    month_max = df.groupby(by=['month']).max()
    month_max.drop(columns=['forc_1_ms', 'forc_2_ms'], inplace=True)
    month_max = month_max.add_suffix('_max')

    # merge dataframes
    df = pd.merge(df, month_avg, how='left', left_on='month', right_on=month_avg.index)
    df = pd.merge(df, month_min, how='left', left_on='month', right_on=month_min.index)
    df = pd.merge(df, month_max, how='left', left_on='month', right_on=month_max.index)

    # drop not needed columns
    df.drop(columns=['datetime_min', 'datetime_max','month'], inplace=True)
    #df.set_index('datetime', inplace=True)
    return df


# calculate the avg, min, max 
# we split where we will split training and testing 
year_1_end = pd.to_datetime('2015-12-31 23:00:00')
split = dataset_c.index[dataset_c['datetime'] == year_1_end][0]
df_year1 = dataset_c.loc[:split,]
df_year1 = add_features(df_year1)

# calculate the rolling avg, min, max and add to list
rolling_values = []
for t in range(split+1, len(dataset_c)+1):
    # limit df 
    df = dataset_c[:t]
    # run algorithm
    df = add_features(df=df)
    # append only latest values
    rolling_values.append(df[-1:].values.tolist()[0])


# convert to list and append to orginial dataframe
rolling_df = pd.DataFrame(rolling_values, columns=list(df_year1.columns))
dataset_c = df_year1.append(rolling_df)

# # Sanity Check (for static and not rolling avg, min, max)
# # This was used to check if the function above works and it did
# def sanity_check(row, column):
#     index = dataset_c.index[row]
#     month = index.month
#     if (month_avg.loc[month, str(column)+'_avg'] == dataset_c.loc[index, str(column)+'_avg'] and
#         month_min.loc[month, str(column)+'_min'] == dataset_c.loc[index, str(column)+'_min'] and
#             month_max.loc[month, str(column)+'_max'] == dataset_c.loc[index, str(column)+'_max']):
#         print('The algorithm worked correct!')
#     else: 
#         print('The algorithm is incorrect, please check manually!')
#         print(month_avg.loc[month, str(column)+'_avg'])
#         print(dataset_c.loc[index, str(column)+'_avg'])

# sanity_check(row=20, column=2)
# sanity_check(row=120, column=13)
# sanity_check(row=1, column=1)

# add month to dataset which is time independent!
dataset_c.set_index('datetime', inplace=True)
dataset_c['month'] = dataset_c.index.month
dataset_c['hours'] = dataset_c.index.hour

# save dataframe
dataset_c.to_csv('data_processed/dataset_c.csv')


### DATASET D: WITH LAGGED VALUES ### ----------------------------------------------------

# copy dataset
dataset_d = dataset_c.copy()

# # create lagged sums
dataset_d['sum_-1'] = dataset_d[cols].sum(axis=1).shift(1)
dataset_d['sum_-2'] = dataset_d[cols].sum(axis=1).shift(2)
dataset_d['sum_-3'] = dataset_d[cols].sum(axis=1).shift(3)
dataset_d['sum_-4'] = dataset_d[cols].sum(axis=1).shift(4)

# create lagged forecasts
dataset_d['forc_-1'] = (dataset_d['forc_1_ms'].shift(1) + dataset_d['forc_2_ms'].shift(1) ) / 2
dataset_d['forc_-2'] = (dataset_d['forc_1_ms'].shift(2) + dataset_d['forc_2_ms'].shift(2) ) / 2
dataset_d['forc_-3'] = (dataset_d['forc_1_ms'].shift(3) + dataset_d['forc_2_ms'].shift(3) ) / 2
dataset_d['forc_-4'] = (dataset_d['forc_1_ms'].shift(4) + dataset_d['forc_2_ms'].shift(4) ) / 2

# create averages for some windows
dataset_d['prod_avg_24h'] = dataset_d['prod_sum'].rolling(window=24).mean()
dataset_d['prod_avg_7d']  = dataset_d['prod_sum'].rolling(window=24*7).mean()
dataset_d['prod_avg_30d'] = dataset_d['prod_sum'].rolling(window=24*30).mean()

# save to csv
dataset_d.to_csv('data_processed/dataset_d.csv')
