import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale = 1.6)
sns.set_palette("binary_r")
sns.set_style('whitegrid')

filename = "DeepQFinancialMarket dealerPosition-vs-skew-spreadsheet.csv"
data = pd.read_csv(filename).iloc[:, 1:]

form = pd.concat([pd.Series(data.iloc[-1, 0::2].values), 
			pd.Series(data.iloc[-1, 1::2].values)], axis=1)

form.columns = ['Market Maker net-long (negative values: net-short)', 'Price Distribution Skew']

_max, _min = form['Market Maker net-long (negative values: net-short)'].max(), form['Market Maker net-long (negative values: net-short)'].min()
scale_d = (_max - form['Market Maker net-long (negative values: net-short)'])/(_max-_min)
scale_u = (form['Market Maker net-long (negative values: net-short)'] - _min)/(_max-_min)
colors = [[c*scale_d.iloc[i] for c in (1, 0.1, 0)] for i in range(len(scale_d))]
for j in range(len(colors)):
	s = scale_d[j]
	if s < 0.5:
		colors[j] = [c*scale_u.iloc[j] for c in (0, 1, 0.1)]

print(colors)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.regplot(data=form, x='Market Maker net-long (negative values: net-short)', y='Price Distribution Skew', ax=ax,
            scatter_kws={'color':colors}, line_kws={'color':'blue'})
plt.show()
