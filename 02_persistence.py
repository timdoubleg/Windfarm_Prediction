import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse, r2_score
import os

warnings.filterwarnings('ignore')

### DATA IMPORT ### ----------------------------------------------------
dataset_a = pd.read_csv('data_processed/dataset_a.csv', index_col='datetime')

# drop last 4 rows as we don't know the actual production for 4h ahead
dataset_a = dataset_a[:-4]


### PERSISTENCE MODEL ### ----------------------------------------------------

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
def return_accuracy(x_test, y_test, name):
    # calculate RMSE, R2, Adj. R2, STDEV, Error Peak
    actual = y_test.values
    cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    prediction = x_test[cols].sum(axis=1)
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

# create folder if doesn't yet exist
if not os.path.exists('results'):
    os.makedirs('results')

# split dataset
x_train, x_test, y_train, y_test = split_test_train(dataset_a)

# calculate accuracy
df = return_accuracy(x_test=x_test, y_test=y_test, name='persistence')

# save to csv
df.to_csv('results/persistence.csv')

