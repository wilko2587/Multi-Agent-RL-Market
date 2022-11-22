import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

agents = (1, 2, 3, 4)
results = []
epsilon = pd.read_csv('e1.csv', index_col=0).iloc[:, 0]

epsilon_zero_idx = epsilon.loc[epsilon<=0.05].index[0]
for n in agents:
    data = pd.read_csv('{}.csv'.format(n), index_col=0).iloc[:, 0]
    data.index = data.index - epsilon_zero_idx
    data = data - data.loc[0]
    results.append(data)
results = pd.concat(results, axis=1)
results.columns = ['trend investor {}'.format(i) for i in agents]
results_pretrain = results.loc[:0]
results_posttrain = results.loc[0:]
results_pretrain['phase'] = "training"
results_posttrain['phase'] = "inferring"
results = pd.concat([results_pretrain, results_posttrain], axis=0)

results.index = pd.cut(results.index, 200)
results.index = [str(i.left) for i in results.index.tolist()]
epsilon.index = pd.cut(epsilon.index, 200)
epsilon.index = [str(i.left) for i in epsilon.index.tolist()]

results = pd.melt(results.reset_index(drop=False), id_vars=['index', 'phase'])

results.columns = ['episode', 'phase', 'agent', 'Cumulative Profit']

fig, axs = plt.subplots(1, figsize=(8, 6))
axs2 = axs.twinx()
palette = "crest" #sns.color_palette("ch:start=.2,rot=-.3")
g = sns.lineplot(data=results, x='episode', y='Cumulative Profit',
             hue='agent',
             style='phase',
             style_order=['inferring', 'training'],
             estimator='mean', ax=axs, #palette='crest',
             palette=palette,
             err_kws={"alpha": 0.0},
             alpha=1,
             legend=True)


epsilon = epsilon.reset_index(drop=False)
epsilon.columns = ['episode', 'epsilon']
epsilon['episode'] = epsilon['episode'].astype('float').astype('int') - epsilon_zero_idx
epsilon = epsilon.groupby('episode').agg('mean').reset_index(drop=False)
epsilon['episode'] = epsilon['episode'].astype('str')
eps_plot = epsilon.plot(ax=axs2, color='black', label='ε')
#axs.set_yscale('log')
h, l = g.get_legend_handles_labels()
h2, l2 = eps_plot.get_legend_handles_labels()
h = h + h2
l = l + l2
g.get_legend().remove()
eps_plot.get_legend().remove()
axs.legend(h, l, loc='upper center', ncols=3)
axs2.set_ylabel("ε")

xlabels = [int(float(i.get_text())) for i in axs.get_xticklabels()[1:]]

xticks = list(range(0, len(xlabels), 10))
xlabels = [xlabels[i] for i in xticks]
axs.set_xticks(ticks=xticks, labels=xlabels, rotation=35, ha='right')
#axs.set_yticks(ticks=[1e-3, 5e-3, 1e-2, 3e-2], labels=[1e-3, 5e-3, 1e-2, 3e-2], rotation=0, ha='right')
axs.set_ylabel('Cumulative Profit')
axs.set_xlabel('Episodes following training completion')
#axs.set_ylim([1e-3, 3e-2])
axs.grid(b=True, which='major', color='black', linewidth=1.0, alpha=0.2, linestyle='--')
axs.grid(b=True, which='minor', color='black', linewidth=0.5, alpha=0.2, linestyle='--')
plt.show()
