import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

results = pd.DataFrame(index=None, columns=['agent {}'.format(i) for i in (6,7,8,9,10)])
for n in results.columns:
    data = pd.read_csv('{}.csv'.format(n.split(' ')[1]), index_col=0).iloc[:, 0]
    results.loc[:, n] = data.values

results.index = pd.cut(results.index, 200)
results.index = [str(i.left) for i in results.index.tolist()]

print(results.index)
fig, axs = plt.subplots(1, figsize=(12, 6))
sns.lineplot(data=results, alpha=0.5, estimator='mean', ax=axs)
plt.xticks(rotation=45, ha='right')
start, end = axs.get_xlim()
#print([float(i.get_text()) for i in axs.get_xticklabels()[1:]])
xlabels = [int(float(i.get_text())) for i in axs.get_xticklabels()[1:]]
#axs.set_xticklabels([str(int(float(i.get_text()))) for i in axs.get_xticklabels()[1:]])
xticks = list(range(0, len(xlabels), 10))
xlabels = [xlabels[i] for i in xticks]
print(xticks)
print(xlabels)
axs.set_xticks(ticks=xticks, labels=xlabels)
#all_labels = [i.get_text() for i in fig.axes[0].axes.get_xticklabels()]
#wanted_idx = [i for i in range(0, len(fig.axes[0].axes.get_xticklabels()), 10)]
#print(wanted_idx)
#print(all_labels)
#print(len(all_labels))
#fig.axes[0].axes.set_xticks(wanted_idx)
#fig.axes[0].axes.set_xticklabels([all_labels[i] for i in wanted_idx])
plt.show()