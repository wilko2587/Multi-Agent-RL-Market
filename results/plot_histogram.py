import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

sns.set(font_scale = 1.6)
sns.set_palette("binary_r")
sns.set_style('whitegrid')
#sns.set(font_scale = 0.7)
#sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})

i = 0
filename = 'returns_market_fragmentation.csv'
data = pd.read_csv(filename, index_col=0)
binwidth = 0.5
print(data)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
sns.histplot(data=data, x='100', ax=axs[0], binwidth=binwidth, color='blue',
             edgecolor="black", linewidth=2, label='Observed', alpha=0.5)

xmin, xmax = -15, 15
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, float(data['100'].mean()), float(data['100'].std())) * len(data['100']) * binwidth
title = "Normal \nmu = %.2f \nstd = %.2f" % (data['100'].mean(), data['100'].std())
sns.lineplot(x=x, y=p, linewidth=2, label=title, ax=axs[0], color='red')

sns.histplot(data=data, x='100', ax=axs[1], binwidth=binwidth, color='blue',
             edgecolor="black", linewidth=2, alpha=0.5)
sns.lineplot(x=x, y=p, linewidth=2, ax=axs[1], color='red')

axs[1].set_yscale('log')
axs[1].set_ylim([1e-3, 1e3])
for ax in axs:
    ax.set_xlim([xmin, xmax])
axs[0].set_xlabel("")
axs[1].set_xlabel("50-tick price change")

axs[0].text(-12, 250, "Linear scale", fontsize=20)
axs[1].text(-12, 100, "Log scale", fontsize=20)
plt.show()