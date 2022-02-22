from pyexpat import model
import warnings
from turtle import title
import matplotlib.pyplot as plt
#plt.rcParams.update({'figure.figsize': (9, 6), 'figure.dpi': 120})
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa import *
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
plt.style.use('fivethirtyeight')

warnings.filterwarnings('ignore')

### DATA IMPORT ### ----------------------------------------------------
dataset_a = pd.read_csv('data_processed/dataset_a.csv', index_col='datetime')
dataset_b = pd.read_csv('data_processed/dataset_b.csv', index_col='datetime')
dataset_c = pd.read_csv('data_processed/dataset_c.csv', index_col='datetime')
dataset_d = pd.read_csv('data_processed/dataset_d.csv', index_col='datetime')
dataset_e = pd.read_csv('data_processed/dataset_e.csv', index_col='datetime')

cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

dataset = dataset_a

df_prods = dataset[cols]
df_forc = dataset[['forc_1_ms', 'forc_2_ms']]

df_prods.index = pd.to_datetime(df_prods.index)
dataset.index = pd.to_datetime(dataset.index)
df_forc.index = pd.to_datetime(df_forc.index)

# drop last values
df_prods.dropna(inplace=True)
dataset.dropna(inplace=True)


### RESMAPLING ### ----------------------------------------------------

df_prods_daily = df_prods.resample(rule='d').sum()
df_prods_weekly = df_prods.resample(rule='w').sum()
df_prods_monthly = df_prods.resample(rule='m').sum()

df_forc_daily = df_forc.resample(rule='d').sum()
df_forc_weekly = df_forc.resample(rule='w').sum()
df_forc_monthly = df_forc.resample(rule='m').sum()

dataset_daily = dataset.resample(rule='d').sum()
dataset_weekly = dataset.resample(rule='w').sum()
dataset_monthly = dataset.resample(rule='m').sum()

### PLOTTING ### ----------------------------------------------------

# First we need to check for stationarity

def plot_all(df, nrows=5, ncols=4, title=title): 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20,12))
    try: 
        for i, ax in enumerate(axes.flatten()):
            data = df[df.columns[i]]
            ax.plot(data, color='red', linewidth=1)
            # Decorations
            ax.set_title(df.columns[i])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)
    except:
        pass

    plt.tight_layout()
    plt.suptitle(title)

plot_all(dataset_daily, nrows=6, ncols=4, title='Daily Sums')
plot_all(dataset_weekly, nrows=6, ncols=4, title='Weekly Sums')
plot_all(dataset_monthly, nrows=6, ncols=4, title='Monthly Sums')

### GRANGER CAUSALITY TEST ### ----------------------------------------------------

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))),columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            
            if verbose:
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value

    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


# run test
gcm = grangers_causation_matrix(dataset, variables=dataset.columns)
print(gcm)
sns.heatmap(gcm, annot=True)


# If a given p-value is < significance level(0.05), then, the corresponding X series(column) causes the Y(row). 
# For example, P-Value of 0.0003 at(row 1, column 2) represents the p-value of the Grangers Causality test for 
# x1 causing y1, which is less that the significance level of 0.05. So, you can reject the null hypothesis 
# and conclude x1 causes y1. Looking at the P-Values in the above table, you can pretty much observe that 
# all the variables(time series) in the system are interchangeably causing each other


### COINTEGRATION TEST ### ----------------------------------------------------

def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary
    When two or more time series are cointegrated, it means they have a long run,
    statistically significant relationship. 
    This is the basic premise on which Vector Autoregression(VAR) models is based on"""
    out = coint_johansen(df, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9),
              ">", adjust(cvt, 8), ' =>  ', trace > cvt)

# run test
cointegration_test(dataset)

### SPLITTING DATASET ### ----------------------------------------------------

nobs = 4
df_train, df_test = dataset[0:-nobs], dataset[-nobs:]

# Check size
print(df_train.shape) 
print(df_test.shape)  

### TESTING FOR STATIONARITY ### ----------------------------------------------------

## Basically three possibilities
# 1. Augmented Dickey-Fuller Test(ADF Test)
# 2. KPSS test
# 3. Philip-Perron test Let’s use the ADF test for our purpose

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")  


# ADF Test on each column
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


### SELECT ORDER FOR VAR MODEL ----------------------------------------------------

# specify model 
model = VAR(df_train)
max_lags = 10

# test various lags 
# To select the right order of the VAR model, we iteratively fit increasing orders of
# VAR model and pick the order that gives a model with least AIC
for i in range(1, max_lags+1):
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic, '\n')

# --> order (1,0,0) is best

def plot_acf_pacf(df, title):
    nrows = len(df.columns)
    ncols =2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20, 60))
    for i in range(0, len(df.columns)):
        data = df[df.columns[i]]
        sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=10, ax=ax[i, 0], title=str(dataset.columns[i]), zero=False, auto_ylims=True)
        sm.graphics.tsa.plot_pacf(data.values.squeeze(), lags=10, ax=ax[i, 1], title=str(dataset.columns[i]), zero=False, auto_ylims=True)
        #plt.show()
    
    plt.tight_layout()
    plt.suptitle(title)

plot_acf_pacf(df=dataset, title='ACF & PACF')

model_fitted = model.fit(1)
model_fitted.summary()

### DIFFERENCING ### ----------------------------------------------------

# Differencing
dataset_diff = dataset.diff()
dataset_diff = dataset_diff[:-5].dropna()

# splitting dataset
nobs = 4
df_train, df_test = dataset_diff[0:-nobs], dataset_diff[-nobs:]

# plot acf and pacf
plot_acf_pacf(df=dataset_diff, title='ACF & PACF')

# make model
model = VAR(dataset_diff)
max_lags = 10

# test various lags
# To select the right order of the VAR model, we iteratively fit increasing orders of
# VAR model and pick the order that gives a model with least AIC
for i in range(1, max_lags+1):
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic, '\n')


model_fitted = model.fit(1)
model_fitted.summary()


### RESIDUALS CHECK ### ----------------------------------------------------

#  If there is any correlation left in the residuals, 
# then, there is some pattern in the time series that is still left 
# to be explained by the model. In that case, the typical course of 
# action is to either increase the order of the model or induce more
#  predictors into the system or look for a different algorithm to model the time series

out = durbin_watson(model_fitted.resid)

for col, val in zip(df_train.columns, out):
    def adjust(val, length=6): return str(val).ljust(length)
    print(adjust(col), ':', round(val, 2))

# The closer it is to the value 2, then there is no significant serial correlation. 
# The closer to 0, there is a positive serial correlation
# The closer it is to 4 implies negative serial correlation


# ---> Most are between 1 - 2, hence we can say we have no significant serial correlation of the residuals


### PREDICT ### ----------------------------------------------------

# Get the lag order
lag_order = model_fitted.k_ar
print(f"lag order of: {lag_order}")  #

# Input data for forecasting
forecast_input = df_train.values[-lag_order:]
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=dataset.index[-nobs:], columns=dataset.columns)


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing (if needed) to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forc'] = df_train[col].iloc[-1] + df_fc[(col)].cumsum()
    return df_fc

# plot forecast vs actual
def plot_forc_act(df_test, df_forecast):
    fig, axes = plt.subplots(nrows=int(len(df_test.columns)/4), ncols=4, dpi=150, figsize=(20,20))
    for i, (col,ax) in enumerate(zip(df_test.columns, axes.flatten())):
        df_forecast[col].plot(legend=True, ax=ax, label='forecast').autoscale(axis='x',tight=True)
        df_test[col][-nobs:].plot(legend=True, ax=ax, label='actual');
        ax.set_title(col)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout();

df_results = invert_transformation(df_train, df_forecast)        

# plot
plot_forc_act(df_test=df_test, df_forecast=df_forecast)


### EVALUATE FORECAST ### ----------------------------------------------------

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})


def evaluate_forecast(df_forecast, df_test):
    dictionary = {}
    for i in df_forecast.columns:
        accuracy_prod = forecast_accuracy(df_forecast[i].values, df_test[i])
        dictionary[i] = accuracy_prod
        # for k, v in accuracy_prod.items():
        #     print(adjust(k), ': ', round(v, 4))
        # print('\n')
    df = pd.DataFrame.from_dict(dictionary, orient='columns', dtype=None)
    return df



### DO EVERYTHING IN A ROLLING WINDOW ### ----------------------------------------------------

beginning_value = 10
results_list = []
for t in range(beginning_value, len(dataset)):
    # print progress
    if t % 100 == 0: 
        print(f"row {t}: {round(t/len(dataset),2)*100} % ")

    # rolling window
    df = dataset[0:t]
    nobs = 4
    df_train, df_test = df[0:-nobs], df[-nobs:]

    # specify model
    model = VAR(df_train)
    max_lags = 10
    
    # VAR(1)
    model_fitted = model.fit(1)
    lag_order = model_fitted.k_ar

    # forecast
    forecast_input = df_train.values[-lag_order:]
    fc = model_fitted.forecast(y=forecast_input, steps=nobs)

    # evaluate forecast
    results = evaluate_forecast(df_forecast=df_forecast, df_test=df_test)
    results = results.loc['rmse', 'prod_sum']
    results_list.append(results)

rmse = np.mean(results_list)

# results_list
# results_mean = pd.concat(results_list).mean()

# dictionary = {}
# for df in results_list:
#     for col in df.columns:
#         if dictionary == {}:
#             dictionary[col] = [df[col].mape]
#             dictionary[col] = [df[col].me]
#             dictionary[col] = [df[col].mae]
#             dictionary[col] = [df[col].mpe]
#             dictionary[col] = [df[col].rmse]
#             dictionary[col] = [df[col].corr]
#             dictionary[col] = [df[col].minmax]
#         else: 
#             dictionary[col].append(df[col].mape)
#             dictionary[col].append(df[col].me)
#             dictionary[col].append(df[col].mae)
#             dictionary[col].append(df[col].mpe)
#             dictionary[col].append(df[col].rmse)
#             dictionary[col].append(df[col].corr)
#             dictionary[col].append(df[col].minmax)

# dictionary[1]


# save results for now
import pickle
# with open("results_list", "wb") as fp:   #Pickling
#     pickle.dump(results_list, fp)

with open("results_list", "rb") as fp:   # Unpickling
    results_list = pickle.load(fp)
