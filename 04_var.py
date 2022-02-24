from pyexpat import model
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa import *
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')

### DATA IMPORT ### ----------------------------------------------------
dataset_a = pd.read_csv('data_processed/dataset_a.csv', index_col='datetime')
dataset_b = pd.read_csv('data_processed/dataset_b.csv', index_col='datetime')
dataset_c = pd.read_csv('data_processed/dataset_c.csv', index_col='datetime')
dataset_d = pd.read_csv('data_processed/dataset_d.csv', index_col='datetime')
dataset_e = pd.read_csv('data_processed/dataset_e.csv', index_col='datetime')

# drop values as we only have actual values until '2017-04-08 17:00:00'
dataset_a = dataset_a[:-4]
dataset_b = dataset_b[:-4]
dataset_c = dataset_c[:-4]
dataset_d = dataset_d[:-4]
dataset_e = dataset_e[:-4]

dataset = dataset_a.copy()
dataset.index = pd.to_datetime(dataset.index)
width, height = 16, 10

### VISUALIZE PROD  ### ----------------------------------------------------

dataset['prod_sum'].plot(figsize=(width, height))

### NORMALIZE  ### ----------------------------------------------------

avgs = dataset.mean()
devs = dataset.std()

for col in dataset.columns:
    dataset[col] = (dataset[col] - avgs.loc[col]) / devs.loc[col]

dataset['prod_sum'].plot(figsize=(width, height))

### FIRST DIFFERENCE TO GET MORE STATIONARY DATA ### ----------------------------------------------------

dataset_diff = dataset.diff().dropna()
dataset_diff['prod_sum'].plot(figsize=(width, height))

### REMOVE HETEROSKEDASTICITY ACROSS MONTHS ### ----------------------------------------------------

month_volatility = dataset_diff.groupby(dataset_diff.index.month).std()
dataset_rm_std = dataset_diff.index.map(lambda d: month_volatility.loc[d.month, ])
dataset_rm_std = pd.DataFrame.from_records(dataset_rm_std)

dataset_homoskedastic = dataset_diff.values / dataset_rm_std.values
names = ['forc_1_ms', 'forc_2_ms', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 'prod_sum']
dataset_homoskedastic = pd.DataFrame.from_records(dataset_homoskedastic, columns=names)
dataset_homoskedastic.index = dataset_diff.index

dataset_homoskedastic['prod_sum'].plot(figsize=(width, height))


### REMOVE YEARLY SEASONALITY  ### ----------------------------------------------------

month_avgs = dataset_homoskedastic.groupby(dataset_homoskedastic.index.month).mean()
dataset_rm_season = dataset_homoskedastic.index.map(lambda d: month_avgs.loc[d.month, ])
dataset_rm_season = pd.DataFrame.from_records(dataset_rm_season)

dataset_nonseasonal = dataset_homoskedastic.values - dataset_rm_season.values
dataset_nonseasonal = pd.DataFrame.from_records(dataset_nonseasonal, columns=names)
dataset_nonseasonal.index = dataset_diff.index

dataset_nonseasonal['prod_sum'].plot(figsize=(width, height))

### PACF AND ACF PLOT  ### ----------------------------------------------------

plot_acf(dataset_nonseasonal['prod_sum'].values.squeeze(), zero=False, auto_ylims=True)
plot_pacf(dataset_nonseasonal['prod_sum'].values.squeeze(), zero=False, auto_ylims=True)


### TEST STATIONARITY ### ----------------------------------------------------
def get_stationarity(timeseries, w=12, h=8):
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # rolling statistics plot
    fig = plt.figure(figsize=(w, h))
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey–Fuller test:
    result = stattools.adfuller(timeseries.values)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


def test_ljung_box(model_fit):
    # prints LB statistic for all lags
    jlung = acorr_ljungbox(model_fit.resid)
    print(jlung)

    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()


### FIT MODEL ### ----------------------------------------------------

model = VAR(dataset_nonseasonal)
model_fit = model.fit(maxlags=30, method='ols', ic='bic', verbose=True)

# --> we get 4 lags as best model
model_fit = model.fit(maxlags=4)
results = pd.DataFrame(model_fit.params['prod_sum'])
results['p_value'] = model_fit.pvalues['prod_sum']
results[results['p_value'] < 0.05]

# model_fit.summary()