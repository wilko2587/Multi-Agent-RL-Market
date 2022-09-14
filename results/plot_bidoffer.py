import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale = 1.6)
sns.set_palette("binary_r")
sns.set_style('whitegrid')

#filename = "DeepQFinancialMarket Kurtosis-vs-smartVolume-spreadsheet.csv"
filename = 'bid-offer.csv'
data = pd.read_csv(filename).iloc[:, 1:].dropna(axis=1)
data.index = ['start px', 'end px', 'mean expectation', 'time averaged price', 'mean abs deviation', 'bid offer', 'kurtosis', 'skew']

data.loc['Deviation from mean expectation'] = (data.loc['mean expectation'] - data.loc['time averaged price']).abs()
#data.loc['mean-squared difference from start'] = (data.loc['start px'] - data.loc['time averaged price']).pow(2)

data = data.transpose()
#data = data.loc[data['kurtosis'] > 2]
#data = data.loc[(data['kurtosis'] < 10) & (data['kurtosis'] > 2)]

fig, axs = plt.subplots(1, 1, figsize=(10, 6))
#sns.scatterplot(data=data, x='% market-maker volume', y='kurtosis', ax=axs, label='market-makers')
sns.regplot(data=data, x='bid offer', y='Deviation from mean expectation', ax=axs, order=2, marker='o', line_kws={'linestyle':'solid'}, color='black')
#sns.scatterplot(data=data, x='volume', y='skew', ax=axs, label='Total')
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
