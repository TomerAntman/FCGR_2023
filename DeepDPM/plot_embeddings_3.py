from torch import load as torch_load
from torch import device as torch_device
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

N = 10
S = 0
val_train = 'train'
Shuff = False
PlotWithUnshuff = True
isFlat = True
nonFlat = '/nonFlat' if not isFlat else ''
n_channels = 3
community_name = f"shuffled_community_{N}_{S}" if Shuff else f"community_{N}_{S}"


npz_files = os.listdir(f"/home/labs/testing/class49/PreProcess/communities/N_{N}/FCGR_files")
assert f"community_S-{S}_N-{N}_{n_channels}Ch.npz" in npz_files, f"npz file not community_S-{S}_N-{N}_{n_channels}Ch.npz not found in path /home/labs/testing/class49/PreProcess/communities/N_{N}/FCGR_files/"

path_to_data = f"/home/labs/testing/class49/DeepDPM/test{nonFlat}/{community_name}/results"

embedding_path = [os.path.join(path_to_data,x) for x in os.listdir(path_to_data) if x.startswith(f"{val_train}_codes")][0]

if Shuff and PlotWithUnshuff:
    labels_path = [os.path.join(f"/home/labs/testing/class49/DeepDPM/test/community_{N}_{S}",x) for x in os.listdir(f"/home/labs/testing/class49/DeepDPM/test/community_{N}_{S}") if x.startswith(f"{val_train}_labels")][0]
else:
    labels_path = [os.path.join(path_to_data,x) for x in os.listdir(path_to_data) if x.startswith(f"{val_train}_labels")][0]
#print(labels_path)

embedding = torch_load(embedding_path, map_location=torch_device('cpu'))
labels = torch_load(labels_path, map_location=torch_device('cpu'))
print('loaded embeddings')
df = pd.DataFrame({'x': embedding[:,0], 'y': embedding[:,1], 'numLabel': labels, 'fullLabel': ""})

data = dict(np.load(f"/home/labs/testing/class49/PreProcess/communities/N_{N}/FCGR_files/community_S-{S}_N-{N}_{n_channels}Ch.npz"))
unique_num_labels = np.unique(labels)
for label in unique_num_labels:
	idx = np.where(data['number_labels']==label)[0][0]
	full_name = data['labels'][idx]
	df['fullLabel'][df.numLabel==label] = full_name
print('collected labels')
df = df.sort_values(by=['fullLabel'])

figFileName = f"UMAP_{community_name}_{val_train}_{n_channels}Channels_PlotUnshuff.png" if PlotWithUnshuff and Shuff else f"UMAP_{community_name}_{val_train}_{n_channels}Channels.png"
#fig1 = plt.figure()
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 10))
#ax1 = fig1.add_subplot(111)
g1 = sns.scatterplot(data=df, x='x', y='y', hue='fullLabel', alpha=0.3, s=18, ax = ax1)
g1.set(xticklabels=[],xticks=[],yticks=[],yticklabels=[])
ax1.set_title('True')
#handles1, labels1 = ax1.get_legend_handles_labels()
ax1.set(xlabel=' ', ylabel=' ')
legend1 = g1.legend(loc='upper center', bbox_to_anchor=(0, -0.1), ncol=1, title='True labels', title_fontsize=20, fontsize=15)
ax1.add_artist(legend1)

g2 = sns.scatterplot(data=df, x='x', y='y', hue='predLabel', alpha=0.3, s=18, ax = ax2)
g2.set(xticklabels=[],xticks=[],yticks=[],yticklabels=[])
ax2.set_title('DeepDPM')
#handles1, labels1 = ax1.get_legend_handles_labels()
ax2.set(xlabel=' ', ylabel=' ')
legend2 = g2.legend(loc='upper center', bbox_to_anchor=(0, -0.1), ncol=1, title='True labels', title_fontsize=20, fontsize=15)
ax2.add_artist(legend2)


#ax1.set_title(f"N={N}, S={S}, {val_train if val_train=='train' else val_train+'idation'} data, {'Flat input' if isFlat else 'Unflat input'}, Channels: {n_channels}")
#sns.move_legend(ax1, "upper left", bbox_to_anchor=(0, -0.01))
fig.savefig(f'{path_to_data}/{figFileName}', dpi=300, bbox_inches='tight')
print(f'saved figure to {path_to_data}/{figFileName}')
