import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set(font_scale = 1.6)
#sns.set_palette("binary_r")
sns.set_style('whitegrid')

def clear(filename):
    data = pd.read_csv(filename, index_col=0)
    data = data.iloc[0:1,0:1]
    data.to_csv(filename)
    return

csv_file = "returns_market_fragmentation0.csv"
#clear(csv_file)
#csv_file = "bidoffer_fragmentation.csv"
#clear(csv_file)
#data_file = pd.read_csv(csv_file, index_col=0)
#data = pd.DataFrame(data = data_file.values.flatten(), columns=['Price Chg']).dropna()

data = pd.read_csv(csv_file, index_col=0)
data['Price Chg'] = data.mean(axis=1)
data['Std'] = data.mean(axis=1)

data['Abs. Price Chg'] = data['Price Chg'].abs()
data['Rank'] = data['Abs. Price Chg'].rank(ascending=False)
data['Log Rank (base 10)'] = np.log10(data['Rank'].values)

f = plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='Log Rank (base 10)', y='Abs. Price Chg', line_kws={'color':'red'})
plt.show()