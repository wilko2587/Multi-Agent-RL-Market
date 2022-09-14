import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale = 1.6)
sns.set_palette("binary_r")
sns.set_style('whitegrid')

#filename = "DeepQFinancialMarket Kurtosis-vs-smartVolume-spreadsheet.csv"
filename = 'divergence_vs_volume.csv'
data = pd.read_csv(filename)
data.index = ['start px', 'end px', 'mean expectation', 'time averaged price', 'market-maker volume', 'value volume', 'trend volume', 'kurtosis', 'skew']
data.loc['volume'] = data.loc['market-maker volume'] + data.loc['value volume'] + data.loc['trend volume']

data.loc['% market-maker volume'] = data.loc['market-maker volume']/data.loc['volume']
data.loc['% value volume'] = data.loc['value volume']/data.loc['volume']
data.loc['% trend volume'] = data.loc['trend volume']/data.loc['volume']

data.loc['Deviation from mean expectation'] = (data.loc['mean expectation'] - data.loc['time averaged price']).abs()
#data.loc['mean-squared difference from start'] = (data.loc['start px'] - data.loc['time averaged price']).pow(2)

data = data.transpose()
data = data.loc[data['kurtosis'] > 2]
#data = data.loc[(data['kurtosis'] < 10) & (data['kurtosis'] > 2)]

#fig, axs = plt.subplots(1, 1, figsize=(10, 6))
#sns.scatterplot(data=data, x='% market-maker volume', y='Deviation from mean expectation', ax=axs, label='market-makers')
#sns.scatterplot(data=data, x='% value volume', y='Deviation from mean expectation', ax=axs, label='value investors')
#sns.scatterplot(data=data, x='% trend volume', y='Deviation from mean expectation', ax=axs, label='trend investors')
#axs.set_xlabel("% of volume")
#
#fig, axs = plt.subplots(1, 1, figsize=(10, 6))
##sns.scatterplot(data=data, x='market-maker volume', y='Deviation from mean expectation', ax=axs, label='market-makers')
##sns.scatterplot(data=data, x='value volume', y='Deviation from mean expectation', ax=axs, label='value investors')
##sns.scatterplot(data=data, x='trend volume', y='Deviation from mean expectation', ax=axs, label='trend investors')
#sns.scatterplot(data=data, x='volume', y='Deviation from mean expectation', ax=axs, label='Total')
#axs.set_xlabel("Volume traded")

fig, axs = plt.subplots(1, 1, figsize=(10, 6))
#sns.scatterplot(data=data, x='% market-maker volume', y='kurtosis', ax=axs, label='market-makers')
sns.regplot(data=data, x='% value volume', y='kurtosis', ax=axs, label='value investors', order=2, marker='o', line_kws={'linestyle':'solid'}, color='black')
sns.regplot(data=data, x='% trend volume', y='kurtosis', ax=axs, label='trend investors', order=2, marker='+', line_kws={'linestyle':'solid'}, color='green')
#sns.scatterplot(data=data, x='volume', y='skew', ax=axs, label='Total')
axs.set_xlabel("% volume traded")
#axs.set_yscale('log')
plt.legend()
plt.show()
#form = pd.concat([pd.Series(data.iloc[-1, 0::4].values),
#			pd.Series(data.iloc[-1, 1::4].values),
#			pd.Series(data.iloc[-1, 2::4].values),
#			pd.Series(data.iloc[-1, 3::4].values)], axis=1)
#
#form.columns = ['Kurtosis', 'smartVolume', 'dealerVolume', 'valueVolume']
#form['Proportion of AI traded volume'] = form['smartVolume']/form.iloc[:, 1:].sum(axis=1)
#print(form)
#sns.regplot(data=form, x='dealerVolume', y='Kurtosis')
#plt.show()
