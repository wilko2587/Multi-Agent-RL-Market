import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

agents = (1, 2, 3, 4, 5)
results = []
for n in agents:
    data = pd.read_csv('{}.csv'.format(n), index_col=0).iloc[:, 0]
    results.append(data)
results = pd.concat(results, axis=1)
results.columns = ['agent {}'.format(i) for i in agents]

results.columns = ['trend investor {}'.format(i) for i in agents]
results.index = pd.cut(results.index, 200)
results.index = [str(i.left) for i in results.index.tolist()]

results = pd.melt(results.reset_index(drop=False), id_vars=['index'])
results.columns = ['episode', 'agent', 'Mean Squared Error of Q-Prediction']

fig, axs = plt.subplots(1, figsize=(8, 6))
palette = "Greys"#"crest" #sns.color_palette("ch:start=.2,rot=-.3")
sns.lineplot(data=results, x='episode', y='Mean Squared Error of Q-Prediction',
             hue='agent',
             style='agent',
             estimator='mean', ax=axs, #palette='crest',
             palette=palette,
             err_kws={"alpha": 0.0},
             alpha=1)

axs.set_yscale('log')
plt.legend(loc='upper right')

xlabels = [int(float(i.get_text())) for i in axs.get_xticklabels()[1:]]

xticks = list(range(0, len(xlabels), 10))
xlabels = [xlabels[i] for i in xticks]
axs.set_xticks(ticks=xticks, labels=xlabels, rotation=35, ha='right')
axs.set_yticks(ticks=[1e-3, 5e-3, 1e-2, 3e-2], labels=[1e-3, 5e-3, 1e-2, 3e-2], rotation=0, ha='right')
axs.set_ylabel('Prediction MSE Loss')
axs.set_xlabel('Episode')
axs.set_ylim([1e-3, 3e-2])
axs.grid(b=True, which='major', color='black', linewidth=1.0, alpha=0.2, linestyle='--')
axs.grid(b=True, which='minor', color='black', linewidth=0.5, alpha=0.2, linestyle='--')
plt.show()
