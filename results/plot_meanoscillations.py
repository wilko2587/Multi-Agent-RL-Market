import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale = 1.6)
sns.set_palette("binary_r")
sns.set_style('whitegrid')

x = pd.read_csv('MarketPricesmean109_disparity20.csv')[['x', 'y']].iloc[0:3000]
_mean = 109

print(x)

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(10, 6))
axs[0].set_ylim([70, 130])
axs[1].set_ylim([70, 130])

x.columns = ['Ticks', 'Price']
x['Mean Expectation'] = 109.
sns.lineplot(data=x, x='Ticks', y='Mean Expectation', color='black', ax=axs[0], label='Mean Investor Expectation')
axs[0].lines[0].set_linestyle("dashed")
sns.lineplot(data=x, x='Ticks', y='Price', ax=axs[0], label='Market Price', color='black', linewidth=0.5)
axs[0].set_ylabel('Price')

y = np.arange(70, 130, 2)
x1 = -1*normal_dist(y, 80, 15) * 0.85
x2 = -1*normal_dist(y, 120, 15) * 1.15
normal_df = pd.DataFrame(data=[y, x1, x2]).transpose()[::-1]
normal_df.columns = ['Investor Expectation Distribution', 'x1', 'frequency']
#sns.barplot(data=normal_df, x='x1', y='Investor Expectation', orient = 'h', #palette="Reds",
#            alpha=0.7, ax=axs[1],
#            order=normal_df['Investor Expectation'], color='white', edgecolor="black")
#sns.barplot(data=normal_df, x='frequency', y='Investor Expectation', orient = 'h', #palette='Blues_r',
#            alpha=0.7, ax=axs[1],
#            order=normal_df['Investor Expectation'], color='lightgray', edgecolor="black")

normal_df.columns = ['Investor Expectation Distribution', 'normal1', 'normal2']
normal_df['mean'] = normal_df[['normal1', 'normal2']].mean(axis=1)
sns.barplot(data=normal_df, x='mean', y='Investor Expectation Distribution', orient = 'h', #palette='Blues_r',
            alpha=0.7, ax=axs[1],
            order=normal_df['Investor Expectation Distribution'], color='lightgray', edgecolor="black")

axs[1].set_xlim([normal_df['mean'].min() - 3, normal_df['mean'].max()])
axs[1].set_xlabel('frequency')
axs[1].yaxis.set_label_position("right")
#axs[1].yaxis.set_ticks_position('none')
axs[1].yaxis.tick_right()
plt.yticks([])
#axs[1].get_yaxis().set_visible(False)
#axs[1].set_xticklabels(axs[1].get_xticklabels())

plt.subplots_adjust(wspace=0.05, hspace=0)

plt.show()