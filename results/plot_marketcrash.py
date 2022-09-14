import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale = 1.6)
sns.set_palette("binary_r")
sns.set_style('whitegrid')

data = pd.read_csv('marketcrash.csv')
data.columns = ['ticks'] + data.columns.tolist()[1:]
data = data.iloc[:, 0:4]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
data = data.melt(id_vars=['ticks'])
sns.lineplot(data=data.loc[data['variable'] != '0'], x='ticks', y='value', style='variable', legend=False, color='gray', ax=ax, alpha=0.5)
sns.lineplot(data=data.loc[data['variable'] == '0'].iloc[:1000], x='ticks', y='value', legend=False, color='black', ax=ax)
sns.lineplot(data=data.loc[data['variable'] == '0'].iloc[1000:], x='ticks', y='value', legend=False, color='darkred', ax=ax)
ax.set_ylabel('Price')
ax.set_xlabel('ticks')

plt.plot([1000, 1000], list(ax.get_ylim()), '--')
plt.text(1100, 105, "Market Crash Instigated")
plt.show()