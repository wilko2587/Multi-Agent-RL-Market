import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

sns.set(font_scale = 1.6)
sns.set_palette("binary_r")
sns.set_style('whitegrid')

def clear(filename):
    data = pd.read_csv(filename, index_col=0, on_bad_lines='skip')
    data = pd.DataFrame()
    data.to_csv(filename)
    return

#csv_file = "returns_market_fragmentation4.csv"
#clear(csv_file)
#csv_file = "bidoffer_fragmentation0.csv"
#clear(csv_file)
#data_file = pd.read_csv(csv_file, index_col=0)
#data = pd.DataFrame(data = data_file.values.flatten(), columns=['Price Chg']).dropna()
csv_stem = 'bidoffer_fragmentation'

all_means = pd.DataFrame(data=None, index=None, columns=None)
N = 6
for n in range(N):
    csv_file = csv_stem + str(n) + '.csv'

    data = pd.read_csv(csv_file, index_col=0)
    cols = data.columns.tolist()
    print(cols)
    vals = pd.Series([int(re.findall("\\d+", c)[0]) for c in cols]).unique().tolist()
    time_series = pd.DataFrame(columns=vals)
    for val in vals:
        bo = data["offer{}".format(val)] - data["bid{}".format(val)]
        time_series.loc[:, val] = bo

    means = time_series.mean(axis=0)
    means = pd.DataFrame(means, columns=['Best offer - best bid'])

    means['Prob. of link'] = vals
    means.set_index('Prob. of link', inplace=True)
    all_means = pd.concat([all_means, means], axis=1)

all_means.columns = ['run {}'.format(i) for i in range(N)]
all_means['mean'] = all_means.mean(axis=1)

f = plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
all_means2 = all_means.reset_index().melt(id_vars=['Prob. of link'])
all_means2.rename(columns={'variable':'run', 'value':'best offer - best bid'}, inplace=True)
p = sns.lineplot(data=all_means2.loc[all_means2.loc[:, 'run'] != 'mean'], x='Prob. of link', y='best offer - best bid', style='run',
             ax=ax, color='dimgray', alpha=1., linewidth=0.5, legend=False)
sns.lineplot(data=all_means2.loc[all_means2.loc[:, 'run'] == 'mean'], x='Prob. of link', y='best offer - best bid',
             ax=ax, color='black', label='Mean')
#ax.set_yscale("log")

f = plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
all_means = -1 * all_means
all_means[all_means < 0] = 0

all_means = all_means.reset_index().melt(id_vars=['Prob. of link'])
all_means = all_means[all_means['value']>0]
all_means.rename(columns={'variable':'run', 'value':'Arbitrage Opportunity'}, inplace=True)
p = sns.lineplot(data=all_means.loc[all_means.loc[:, 'run'] != 'mean'], x='Prob. of link', y='Arbitrage Opportunity', style='run',
             ax=ax, color='dimgray', alpha=1., linewidth=0.5, legend=False)
sns.lineplot(data=all_means.loc[all_means.loc[:, 'run'] == 'mean'], x='Prob. of link', y='Arbitrage Opportunity',
             ax=ax, color='black', label='Arbitrage Opportunity')
#ax.set_yscale("log")
plt.show()