import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from yellowbrick.regressor import residuals_plot, prediction_error
import os

warnings.filterwarnings('ignore')

### DATA IMPORT ### ----------------------------------------------------
dataset_a = pd.read_csv('data_processed/dataset_a.csv', index_col='datetime')
dataset_b = pd.read_csv('data_processed/dataset_b.csv', index_col='datetime')
dataset_c = pd.read_csv('data_processed/dataset_c.csv', index_col='datetime')
dataset_d = pd.read_csv('data_processed/dataset_d.csv', index_col='datetime')


### PERSISTENCE MODEL ### ----------------------------------------------------

# drop NA where we only have the values of the forecast but not production history
dataset_a.dropna(inplace=True)
dataset_b.dropna(inplace=True)
dataset_c.dropna(inplace=True)
dataset_d.dropna(inplace=True)

# drop last 4 rows as we don't know the actual production for 4h ahead
dataset_a = dataset_a[:-4]
dataset_b = dataset_b[:-4]
dataset_c = dataset_c[:-4]
dataset_d = dataset_d[:-4]

# sum up the current production and use this a prediction
def return_accuracy(dataset, name):
    dataset = dataset_a
    cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    prediction = dataset[cols].sum(axis=1)
    actual = dataset['prod_sum']

    # calculate RMSE, R2, Adj. R2, STDEV, Error Peak
    MSE = mse(actual, prediction)
    RMSE = round(np.sqrt(MSE), 4)
    R2 = round(r2_score(actual, prediction), 4)
    n = len(actual)
    p = len(dataset.columns)-1
    R2_ADJ = round(1-(1-R2)*(n-1)/(n-p-1), 4)
    STD_DEV = round(np.std(actual), 4)

    result_dict = {name: {'RMSE': RMSE, 'R2': R2, 'R2_ADJ': R2_ADJ, 'STD_DEV': STD_DEV}}
    return pd.DataFrame.from_dict(result_dict)


# create folder if doesn't yet exist
if not os.path.exists('results'):
    os.makedirs('results')


df = return_accuracy(dataset=dataset_a, name='dataset_a')
df.to_csv('results/persistence.csv')
