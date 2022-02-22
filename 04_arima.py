from operator import mod
from pyexpat import model
import warnings
from math import prod
from turtle import title
import matplotlib.pyplot as plt
#plt.rcParams.update({'figure.figsize': (9, 6), 'figure.dpi': 120})
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa import *
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')

### DATA IMPORT ### ----------------------------------------------------
df_merged = pd.read_csv('data_processed/data_merged.csv', index_col='datetime')
df_merged.index = pd.to_datetime(df_merged.index)

prod_1 = df_merged['1']
prod_2 = df_merged['2']
prod_3 = df_merged['3']
prod_4 = df_merged['4']
prod_5 = df_merged['5']
prod_6 = df_merged['6']
prod_7 = df_merged['7']
prod_8 = df_merged['8']
prod_9 = df_merged['9']
prod_10 = df_merged['10']
prod_11 = df_merged['11']
prod_12 = df_merged['12']
prod_13 = df_merged['13']
prod_14 = df_merged['14']
prod_15 = df_merged['15']
prod_16 = df_merged['16']
prod_17 = df_merged['17']
prod_18 = df_merged['18']
prod_19 = df_merged['19']
prod_20 = df_merged['20']

all_prods = [prod_1, prod_2, prod_3, prod_4, prod_5, prod_6, prod_7, prod_8, prod_9, prod_10,
             prod_11, prod_12, prod_13, prod_14, prod_15, prod_16, prod_17, prod_18, prod_19, prod_20]

### OPTIONAL: DELETE NEGATIVE DATA ### ----------------------------------------------------

def DelNegDays(df):
    # resample to days
    df_days = df.resample(rule='d').sum()
    # drop all daily negative values
    df_neg = df_days[df_days < 0].dropna()
    # fetch all dates where we have negative production
    neg_dates = df_neg.index.date
    df_cleaned = df
    # delete data in the hourly data based on the daily data
    for date in neg_dates:
        df_cleaned = df_cleaned.loc[df_cleaned.index.date != date]
    
    # print how many rows are deleted
    print(f"{len(df) - len(df_cleaned)} rows have been deleted")

    return df_cleaned

#prod_1 = DelNegDays(prod_1)


##### AD FULLER TEST FOR STATIONARITY #### ----------------------------------------------------------------------------------------------------

# Augemented Dickey-Fuller Test
df = prod_1
df_diff = prod_1.diff().dropna()
df_roll = prod_1 - prod_1.rolling(window=24).mean().fillna(0)

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

get_stationarity(df)
get_stationarity(df.resample(rule='d').sum())
get_stationarity(df.resample(rule='w').sum())

get_stationarity(df_diff)
get_stationarity(df_diff.resample(rule='d').sum())
get_stationarity(df_diff.resample(rule='w').sum())

get_stationarity(df_roll)
get_stationarity(df_roll.resample(rule='d').sum())
get_stationarity(df_roll.resample(rule='w').sum())

# The more negative this statistic, the more likely we are to reject the null hypothesis of non-stationarity (hence indicating that we have a stationary dataset).
# Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.


##### FIND ARIMA MODEL ORDER #### ----------------------------------------------------------------------------------------------------

print(plot_acf(df, lags=10, title='ACF Absolute'))
print(plot_pacf(df, lags=10, title='PACF Absolute'))

print(plot_acf(df_diff, lags=10, title='ACF Differences'))
print(plot_pacf(df_diff, lags=10, title='PACF Differences'))

### ARIMA model ##### ----------------------------------------------------------------------------------------------------


def arima_model(endog, order):
    # train and print the model
    model = ARIMA(endog=endog, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

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

# Ljung Box: since p-value is above 0.05.94 for AR(2) we cannot reject the null hypothesis that residuals are white noise
# Jarque-Bera: Prob(JB) is at 0 so we know we should not be dealing with heteroskedasticity

### PREDICTIONS ##### ----------------------------------------------------------------------------------------------------

def arima_predictions_plot(model_fit, endog, dynamic=0):
    # In-sample one-step-ahead predictions
    predict = model_fit.get_prediction(dynamic)
    predict_ci = predict.conf_int()

    # when value is 0 this gives a huge confidence interval resulting in problems later when plotting
    predict_ci['lower 1'][0] = 0
    predict_ci['upper 1'][0] = 0

    # Graph
    fig, ax = plt.subplots(figsize=(9,4))
    npre = 4
    ax.set(title='Forecast Electricity Production', xlabel='Date', ylabel='MW')

    # Plot data points
    endog.plot(ax=ax, style='o', label='Observed')

    # Plot predictions
    predict.predicted_mean.plot(ax=ax, style='r--', label='One-step-ahead forecast')
    ci = predict_ci
    ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

    legend = ax.legend(loc='lower right')


def arima_predictions(model_fit, endog, dynamic=0):
    # In-sample one-step-ahead predictions
    predict = model_fit.get_prediction(dynamic)
    predict_ci = predict.conf_int()
    predict_error = predict.predicted_mean - endog
    mape = round(np.abs(np.mean(predict_error)), 6)
    return predict_ci, predict_error, mape


### RUN CODE WITH DIFFERENC MODELS AND COMPARE ##### ----------------------------------------------------------------------------------------------------

# specify which model to use
order = (1, 0, 0)
endog = df_merged['1']
model_fit = arima_model(endog=endog, order=order)
test_ljung_box(model_fit=model_fit)
arima_predictions_plot(model_fit=model_fit, endog=endog)
predict_ci, predict_error, mape = arima_predictions(model_fit=model_fit, endog=endog)

# specify which model to use
order = (1, 0, 0)
endog_2 = df_diff
model_fit_2 = arima_model(endog=endog_2, order=order)
test_ljung_box(model_fit=model_fit_2)
arima_predictions_plot(model_fit=model_fit, endog=endog)
predict_ci_2, predict_error_2, mape_2 = arima_predictions(model_fit=model_fit, endog=endog)

# specify which model to use
order = (1, 1, 0)
endog = df
model_fit = arima_model(endog=endog, order=order)
test_ljung_box(model_fit=model_fit)
arima_predictions_plot(model_fit=model_fit, endog=endog)
predict_ci_3, predict_error_3, mape_3 = arima_predictions(model_fit=model_fit, endog=endog)


### MULTIPLE STEPS AHEAD ----------------------------------------------------------------------------------------------------

order = (1, 0, 0)
endog = df_diff
ahead = len(endog)-50


mod = sm.tsa.statespace.SARIMAX(endog, order=(1, 0, 0))
res = mod.fit()
print(res.summary())
res = mod.filter(res.params)

# In-sample one-step-ahead predictions
predict = res.get_prediction()
predict_ci = predict.conf_int()

# Dynamic predictions
predict_dy = res.get_prediction(dynamic=ahead)
predict_dy_ci = predict_dy.conf_int()

# Graph
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='Electricity Forecast', xlabel='Date', ylabel='MW Difference')

# Plot data points
endog.iloc[ahead:].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.iloc[ahead:].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.iloc[ahead:]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
predict_dy.predicted_mean.iloc[ahead:].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
ci = predict_dy_ci.iloc[ahead:]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

legend = ax.legend(loc='lower right')





### PREDICTIONS (manual) ##### ----------------------------------------------------------------------------------------------------

# order = (1,0,0)
# endog = df_diff

# model = ARIMA(endog=endog, order=order)
# model_fit = model.fit()

# # In-sample one-step-ahead predictions
# predict = model_fit.get_prediction()
# predict_ci = predict.conf_int()
# predict_ci['lower 1'][0] = 0
# predict_ci['upper 1'][0] = 0

# # Graph
# fig, ax = plt.subplots(figsize=(9,4))
# npre = 4
# ax.set(title='Forecast Electricity Production', xlabel='Date', ylabel='MW')

# # Plot data points
# endog.plot(ax=ax, style='o', label='Observed')

# # Plot predictions
# predict.predicted_mean.plot(ax=ax, style='r--', label='One-step-ahead forecast')
# ci = predict_ci
# ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

# legend = ax.legend(loc='lower right')


### Plot Prediction error ###

# endog = df
# order = (1, 1, 0)

# predict = model_fit.get_prediction()
# predict_ci = predict.conf_int()

# # Dynamic predictions
# predict_dy = model_fit.get_prediction(dynamic=10000)
# predict_dy_ci = predict_dy.conf_int()

# # Graph
# fig, ax = plt.subplots(figsize=(9, 4))
# npre = 4
# ax.set(title='Forecast error', xlabel='Date', ylabel='Forecast - Actual')

# # In-sample one-step-ahead predictions and 95% confidence intervals
# predict_error = predict.predicted_mean - endog
# predict_error.plot(ax=ax, label='One-step-ahead forecast')
# ci = predict_ci.copy()
# ci.iloc[:, 0] -= endog
# ci.iloc[:, 1] -= endog
# ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='b', alpha=0.1)

# # Dynamic predictions and 95% confidence intervals
# predict_dy_error = predict_dy.predicted_mean - endog
# predict_dy_error.plot(ax=ax, style='r', label='Dynamic forecast')
# ci = predict_dy_ci.copy()
# ci.iloc[:, 0] -= endog
# ci.iloc[:, 1] -= endog
# ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='r', alpha=0.1)

# legend = ax.legend(loc='lower left')
# legend.get_frame().set_facecolor('w')

# ### MSE of model ###
# mse = round(sum(predict_error)/len(predict_error), 6)
# mse_dy = round(sum(predict_dy_error)/len(predict_dy_error), 6)
# print(f"MAPE of {mse} for one-step ahead model {str(order)}")
# print(f"MAPE of {mse_dy} for dynamic model {str(order)}")

### TEST ##### ----------------------------------------------------------------------------------------------------

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]   # corr
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape': mape, 'me': me, 'mae': mae,
            'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
            'corr': corr, 'minmax': minmax})

