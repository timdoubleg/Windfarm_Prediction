import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
import os
import json
import datetime as dt
plt.rcParams["figure.figsize"] = (10, 7)

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

### DATA IMPORT ### ----------------------------------------------------
dataset_a = pd.read_csv('data_processed/dataset_a.csv', index_col='datetime')
dataset_b = pd.read_csv('data_processed/dataset_b.csv', index_col='datetime')
dataset_c = pd.read_csv('data_processed/dataset_c.csv', index_col='datetime')
dataset_d = pd.read_csv('data_processed/dataset_d.csv', index_col='datetime')

# convert index to datetime again
dataset_a.index = pd.to_datetime(dataset_a.index)
dataset_b.index = pd.to_datetime(dataset_b.index)
dataset_c.index = pd.to_datetime(dataset_c.index)
dataset_d.index = pd.to_datetime(dataset_d.index)

# drop values as we only have actual values until '2017-04-08 17:00:00'
dataset_a = dataset_a[:-4]
dataset_b = dataset_b[:-4]
dataset_c = dataset_c[:-4]
dataset_d = dataset_d[:-4]


### SPLIT TEST AND TRAINING DATA ### ----------------------------------------------------

def split_test_train(dataset, split = 0.75):
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


### TUNE PARAMETERS ### ----------------------------------------------------

# as this is computationally heavy we only do this once for dataset_a
while True:
    try:
        userInput = input(
            'Do you want to perform Grid Search? (Warning this takes a long time!)\n')

        if userInput == 'y':
            print('beginning Grid Search')

            # A parameter grid for XGBoost
            params = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
            }

            # set the regressor
            reg = xgb.XGBRegressor(learning_rate=0.05,
                                n_estimators=100,
                                objective='reg:squarederror',
                                nthread=16
                                )

            # Time Series Split
            tscv = TimeSeriesSplit(n_splits=6, gap=1)
            tscv.split(x_train)

            # find best values
            gsearch = GridSearchCV(estimator=reg,
                                cv=tscv,
                                param_grid=params,
                                verbose=3,  # the computation time for each fold and parameter candidate is displayed;
                                scoring='neg_root_mean_squared_error',
                                n_jobs=-1)

            # run the grid search
            grid_res = gsearch.fit(x_train.values, y_train)

            # extract best params
            best_params = grid_res.best_params_

            # save params to results
            best_params
            best_file = open("results/xgb_best_params.json", "w")
            json.dump(best_params, best_file)
            best_file.close()

            # break the loop
            break

        elif userInput == 'n':
            print('skipping Grid Search')
            break
        else:
            print('please input either "y" or "n"')
    except:
        print('please input either "y" or "n"')


### TRAIN THE MODEL ### ----------------------------------------------------

# import best parameters
params_file = open("results/xgb_best_params.json", "r")
best_params = params_file.read()
best_params = json.loads(best_params)
print(best_params)

def run_xgb(x_train, y_train, x_test, y_test):
    # run the xgboost regression
    reg = xgb.XGBRegressor(
                            n_estimators=1000, 
                            max_depth=best_params['max_depth'],  # This parameter identifies the maximum tree depth for base learners.
                            min_child_weight=best_params['min_child_weight'],
                            eta=0.1,
                            subsample=best_params['subsample'], #This parameter defines the subsample ratio of the training dataset.
                            colsample_bytree=best_params['colsample_bytree'], #This parameter defines the subsample ratio of the training dataset.
                            random_state=10, # Controls the randomness involved in creating tress. You may use any integer.
                            learning_rate = 0.02, 
                            gamma=best_params['gamma'],                        

                            booster='gbtree',
                            objective='reg:squarederror',
                            eval_metric='rmse', 
                            njobs=8  # cores of the processor for doing parallel computations to run XGBoost.
                            )
                        
    # train regressor with early stopping rule
    reg.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=50,
            verbose=True)  

    print(f"Best iteration: {reg.best_iteration}")
    print(f"Best RMSE: {reg.best_score}")

    return reg


### SHOW RESULTS OF THE MODEL ### ----------------------------------------------------

# plot learning curve 
def plot_learncurve(model, name):
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    ax.set_ylabel('RMSE')
    ax.set_xlabel('iteration')
    ax.set_title('XGB RMSE (' + name + ')')
    plt.tight_layout()
    fig.savefig('plots/' + name + '_learningcurve.png')


# plot prediction vs actual
def plot_scatter(model, x_test, y_test, name):
    prediction = (model.predict(x_test))
    actual = pd.DataFrame(y_test)
    df = actual
    df['prediction'] = prediction
    fig, ax = plt.subplots()
    ax.scatter(df['prediction'], df['prod_sum'])
    plt.figure(figsize=(10, 6))
    plt.title('Scatter Actual vs. Prediction (' + name + ')')
    plt.xlabel('prediction')
    plt.ylabel('actual')
    plt.tight_layout()

    plt.savefig('plots/' + name + '_scatter.png')

# plot prediction vs actual
def plot_pred_act(model, x_test, y_test, name):
    prediction = (model.predict(x_test))
    actual = pd.DataFrame(y_test)
    df = actual
    df['prediction'] = prediction
    df.index = pd.to_datetime(df.index)
    ax = df.plot(title='Actual vs. Prediction (' + name + ')', style='-', figsize=(10, 7))
    datemin = dt.date(df.index.min().year, df.index.min().month, df.index.min().day)
    datemax = dt.date(df.index.max().year, df.index.max().month, df.index.max().day+1)
    ax.set_xlim(datemin, datemax)
    ax.legend()
    ax.set_xlabel('datetime')
    ax.set_ylabel('MW')
    plt.savefig('plots/' + name + '_predvsact.png')

# plot scatterplot
def plot_scatter_pred_act(model, x_test, y_test, name):
    prediction = (model.predict(x_test))
    actual = pd.DataFrame(y_test)
    df = actual
    df['prediction'] = prediction
    df.index = pd.to_datetime(df.index)
    scatter_plot = df.plot.scatter('prod_sum', 'prediction', alpha=0.5)
    plt.xlabel('actual')
    plt.ylabel('prediction')
    plt.title(('Scatterplot (' + name + ')'))
    plt.savefig('plots/' + name + '_scatter.png')
    return scatter_plot


# return accuracy metrics for model
def return_accuracy(model, x_test, y_test, name):
    # calculate RMSE, R2, Adj. R2, STDEV, Error Peak
    actual = y_test.values
    prediction = model.predict(x_test)
    MSE = mse(actual, prediction)
    RMSE = round(np.sqrt(MSE), 4)
    R2 = round(r2_score(actual, prediction), 4)
    n = len(actual)
    p = len(x_test.columns)-1
    R2_ADJ = round(1-(1-R2)*(n-1)/(n-p-1), 4)
    # STD_DEV = round(np.std(actual), 4)

    result_dict = {name: {'RMSE': RMSE, 'R2': R2, 'R2_ADJ': R2_ADJ}}
    result_df = pd.DataFrame.from_dict(result_dict)

    # create folder if not yet created
    if not os.path.exists('results'):
        os.makedirs('results')

    # save as csv
    result_df.to_csv('results/' + name + '.csv')
    return result_df

### RUN FOR ALL DATASETS ### ----------------------------------------------------

# define datsets
names = ['dataset_a', 'dataset_b', 'dataset_c', 'dataset_d']
datasets = [dataset_a, dataset_b, dataset_c, dataset_d]

# run loop
for i in range(0, len(names)):
    dataset = datasets[i]
    name = names[i]
    # split
    x_train, x_test, y_train, y_test = split_test_train(dataset)
    # run xgb
    model = run_xgb(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    # plot learning curve
    plot_learncurve(model=model, name=name)
    # plot importance
    ax = plot_importance(model, max_num_features=10, title=('Feature importance (' + name + ')'))
    ax.figure.tight_layout()
    ax.figure.savefig('plots/' + name + '_featureimp.png')
    # plot predicted vs actual
    plot_pred_act(x_test=x_test, y_test=y_test, name=name, model=model)
    # plot scatter
    plot_scatter_pred_act(x_test=x_test, y_test=y_test, name=name, model=model)
    # fetch result metrics
    returns = return_accuracy(x_test=x_test, y_test=y_test, name=name, model=model)




