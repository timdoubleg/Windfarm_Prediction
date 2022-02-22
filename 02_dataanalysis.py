import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


### DATA IMPORT ### ----------------------------------------------------
df_prods = pd.read_csv('data_processed/production.csv', index_col='datetime')
df_merged = pd.read_csv('data_processed/data_merged.csv', index_col='datetime')
df_prods.index = pd.to_datetime(df_prods.index)
df_merged.index = pd.to_datetime(df_merged.index)
df_merged.dropna(inplace=True)

prod_data = pd.read_csv('data_processed/production.csv')


# ------------------------------------------------------------------------
### DATA ANALYSIS ### ----------------------------------------------------
# ------------------------------------------------------------------------

### SCATTERPLOTS ### ----------------------------------------------------

# plot wind power vs power production
plt.scatter(df_merged['forc_1_ms'], df_merged['1'], alpha=0.2)
plt.gca().update(dict(title='Wind vs Power Production (Forc 1 vs Plant 1)', xlabel='m/s', ylabel='MW per hour'))

# calculate sums and average for plotting
prod_sum = prod_data.sum(axis=1)
prod_avg = prod_data.mean(axis=1)
forc_avg = (df_merged['forc_1_ms'] + df_merged['forc_2_ms']) / 2

# scatterplot avg vs sum
plt.figure(figsize=(10, 6))
plt.scatter(forc_avg, prod_avg, alpha=0.2)
plt.gca().update(dict(title='Wind vs Power Production (Avg Forc vs Avg Plants)', xlabel='m/s', ylabel='MW per hour'))

# pairplot
sns.pairplot(df_merged)
