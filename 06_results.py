import pandas as pd

# read csvs
results_a = pd.read_csv('results/dataset_a.csv', names=['metric', 'dataset_a'], header=0)
results_b = pd.read_csv('results/dataset_b.csv', names=['metric', 'dataset_b'], header=0)
results_c = pd.read_csv('results/dataset_c.csv', names=['metric', 'dataset_c'], header=0)
results_d = pd.read_csv('results/dataset_d.csv', names=['metric', 'dataset_d'], header=0)
results_p = pd.read_csv('results/persistence.csv', names=['metric', 'persistence'], header=0)
results_ols = pd.read_csv('results/OLS.csv', names=['metric', 'OLS'], header=0)

# merge csvs
df = pd.merge(results_a, results_b, how='left')
df = pd.merge(df, results_c, how='left')
df = pd.merge(df, results_d, how='left')
df = pd.merge(df, results_ols, how='left')
df = pd.merge(df, results_p, how='left')

def get_improvement(df_imp, df_benchmark):
    # calculate % improvements
    for col in df_merged.columns:
        # for R2 and R2 ADJ
        df_imp[col][0] = (df_merged[col][0]-df_benchmark[0]) / (df_merged[col][0])
        df_imp[col][1] = (df_merged[col][1]-df_benchmark[1]) / (df_merged[col][1])

        # for RMSE
        df_imp[col][2] = ((df_benchmark[2]-df_merged[col][2]) / df_benchmark[2])

    return df_imp

# improvement
df_merged = df[['OLS', 'dataset_a', 'dataset_b', 'dataset_c', 'dataset_d']] 
df_per = df['persistence']
df_imp = df_merged.copy()
df_imp_per = get_improvement(df_imp=df_imp, df_benchmark=df_per)
df_imp_per.index = ['R2', 'R2_ADJ', 'RMSE']

# improvement
df_merged = df[['dataset_a', 'dataset_b', 'dataset_c', 'dataset_d']]
df_ols = df['OLS']
df_imp = df_merged.copy()
df_imp_ols = get_improvement(df_imp=df_imp, df_benchmark=df_ols)
df_imp_ols.index = ['R2', 'R2_ADJ', 'RMSE']

df.set_index('metric', inplace=True)

# print tables
print('Results Absolute\n', df)
print('\n')
print('Improvements to Persistence Model\n', df_imp_per)
print('\n')
print('Improvements to OLS\n', df_imp_ols)

# save to csv
df.to_csv('results/results_abs.csv', index=True)
df_imp_per.to_csv('results/results_perc_per.csv', index=True)
df_imp_ols.to_csv('results/results_perc_ols.csv', index=True)


# hourly production for comparison
dataset_a = pd.read_csv('data_processed/dataset_a.csv', index_col='datetime')
dataset_a['prod_sum'].mean()
