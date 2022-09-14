import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sns.set_palette("binary_r")
#sns.set_style('whitegrid')
sns.set(font_scale = 0.7)
sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})

N = 1
x = np.logspace(1e-5, 0.5, 50) - 1
i = 0
filename = 'returns_market_fragmentation' + str(i) + '.csv'
data = pd.read_csv(filename, index_col=0)
cols = data.columns
results = pd.DataFrame(index=x[:-1], columns=None)
for col in cols:
    full_series = []
    for i in range(0, N):
        filename = 'returns_market_fragmentation' + str(i) + '.csv'
        data = pd.read_csv(filename, index_col=0)
        px = [100]
        for i in range(len(data)):
            px.append(data.iloc[i].loc[col])

        series = data.loc[:, col]
        series_ret = series.divide(pd.Series(px), axis=0)
        full_series = full_series + series_ret.values.tolist()
    count, division = np.histogram(full_series, bins = x)
    try:
        results.loc[:, col] = count/sum(count)
    except:
        results.loc[:, col] = 0

fig, axs = plt.subplots(3, 4, figsize=(10, 6))
axs = axs.flatten().tolist()
print(cols)
for col in cols[:12]:
    ax = axs.pop(0)
    ax.set_title('prob-of-link {}%'.format(col))
    data = results.loc[:, col]
    data = data.loc[::-1].cumsum().loc[::-1]
    data = data.reset_index()
    data.columns = ['price movement', 'P(|price movement| > x)']
    if sum(data['P(|price movement| > x)']) > 0:
        sns.lineplot(data=data.loc[data.loc[:, 'P(|price movement| > x)'] > 0], x = 'price movement', y='P(|price movement| > x)', color='black', linewidth=0.5, ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e-3, 10])
        ax.set_ylim([1e-3, 1])
plt.tight_layout()
plt.show()