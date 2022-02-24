import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as mse, r2_score
import os

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

### DATA IMPORT ### ----------------------------------------------------

dataset_a = pd.read_csv('data_processed/dataset_a.csv', index_col='datetime')
dataset_b = pd.read_csv('data_processed/dataset_b.csv', index_col='datetime')
dataset_b2 = pd.read_csv('data_processed/dataset_b2.csv', index_col='datetime')
dataset_c = pd.read_csv('data_processed/dataset_c.csv', index_col='datetime')
dataset_d = pd.read_csv('data_processed/dataset_d.csv', index_col='datetime')

# convert index to datetime again
dataset_a.index = pd.to_datetime(dataset_a.index)
dataset_b.index = pd.to_datetime(dataset_b.index)
dataset_b2.index = pd.to_datetime(dataset_b2.index)
dataset_c.index = pd.to_datetime(dataset_c.index)
dataset_d.index = pd.to_datetime(dataset_d.index)

# drop values as we only have actual values until '2017-04-08 17:00:00'
dataset_a = dataset_a[:-4]
dataset_b = dataset_b[:-4]
dataset_b2 = dataset_b2[:-4]
dataset_c = dataset_c[:-4]
dataset_d = dataset_d[:-4]

### RUN LINEAR REGRESSION ### ----------------------------------------------------

# split test and train
def split_test_train(dataset, split=0.75):
    # split dependent and independent variables
    data = dataset.drop(columns=['prod_sum']).copy()
    label = dataset['prod_sum']

    # split training and testing data
    length = int(split*len(data))

    # add +4h gap so we do have no time bias
    gap = 4
    length_test = length+gap

    x_train = data.iloc[:length, ]
    x_test = data.iloc[length_test:, ]
    y_train = label.iloc[:length, ]
    y_test = label.iloc[length_test:, ]

    # split train and test
    return x_train, x_test, y_train, y_test

# return accuracy metrics for model
def return_accuracy(model, x_test, y_test, name):
    # calculate RMSE, R2, Adj. R2, STDEV, Error Peak
    actual = y_test.values
    cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    prediction = model.predict(x_test)
    MSE = mse(actual, prediction)
    RMSE = round(np.sqrt(MSE), 4)
    R2 = round(r2_score(actual, prediction), 4)

    # as we have no predictors R2 = R2 ADJ
    R2_ADJ = R2
    # STD_DEV = round(np.std(actual), 4)

    result_dict = {name: {'RMSE': RMSE, 'R2': R2, 'R2_ADJ': R2_ADJ}}
    result_df = pd.DataFrame.from_dict(result_dict)

    # create folder if not yet created
    if not os.path.exists('results'):
        os.makedirs('results')

    # save as csv
    result_df.to_csv('results/' + name + '.csv')
    return result_df

def average_df(data_x): 
    # average forc 1 & 2 and sum up current production
    x = pd.DataFrame(data_x[['forc_1_ms', 'forc_2_ms']].sum(axis=1) / 2, columns=['forecast'])
    cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    prod_sum_today = data_x[cols].sum(axis=1)
    x['prod_sum'] = prod_sum_today
    return x

# to avoid multicollinearity we sum up production and average forecast
x_train, x_test, y_train, y_test = split_test_train(dataset_a)
x_train = average_df(x_train)

# Train linear regression 
mod = sm.OLS(y_train, x_train)
regr = mod.fit()
print(regr.summary())

# Predict linear regression
x_test = average_df(x_test)
return_accuracy(model=regr, name='OLS', x_test=x_test, y_test=y_test)






