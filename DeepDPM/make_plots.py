import os

import numpy as np
import sys
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

def get_colors(n, rev=False):
    ColorPalette = []
    variations = ["bright", "colorblind", "pastel", "muted", "dark"]
    if n>54:
        variations = ["bright", "colorblind","deep", "pastel", "muted", "dark"]
    if rev:
        variations = variations[::-1].copy()
    for var in variations:
        added = sns.color_palette(var)[:7] + sns.color_palette(var)[8:]
        ColorPalette += added

    colors_chosen = ColorPalette[:n]
    return colors_chosen

def main(file_path):
    # get N, S, n_channels from file_path.
    N = int(file_path.split("/N")[1].split("_")[0])
    S = int(file_path.split(f"N{N}_S")[1].split("_")[0])
    n_channels = int(file_path.split(f"N{N}_S{S}_")[1].split("Ch")[0])


    data = np.load(file_path)
    X_embedded = TSNE(n_components=2, learning_rate='auto', metric='cosine').fit_transform(data['features'])
    full_data = dict(np.load(
        f"/home/labs/zeevid/tomerant/FCGR_project/PreProcess/communities/N_{N}/FCGR_files/community_S-{S}_N-{N}_{n_channels}Ch.npz"))
    df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'number_labels': data['labels'],
                       'predLabels': data['predLabels']})

    df['fullLabel']=-1
    unique_num_labels = np.unique(data['labels'])
    for label in unique_num_labels:
         idx = np.where(full_data['number_labels']==label)[0][0]
         full_name = full_data['labels'][idx]
         df['fullLabel'][df.number_labels==label] = full_name
    print('collected labels')
    df = df.sort_values(by=['fullLabel'])

    #path_to_save= "/home/labs/testing/class49/All_results/DeepDPM"
    path_to_save="/home/labs/zeevid/tomerant/FCGR_project/All_results/DeepDPM"
    if "LinAE" in file_path:
        figFileName = file_path.split("LinAE/")[-1].split(".npz")[0] + ".jpg"
    else:
        figFileName = file_path.split("ConvAE/")[-1].split(".npz")[0] + ".jpg"
    pallette1 = get_colors(len(np.unique(df['fullLabel'])), rev=False)
    pallette2 = get_colors(len(np.unique(df['predLabels'])), rev=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
      # ax1 = fig1.add_subplot(111)
    g1 = sns.scatterplot(data=df, x='x', y='y', hue='fullLabel', alpha=0.3, s=18, ax=ax1, palette=pallette1)
    g1.set(xticklabels=[], xticks=[], yticks=[], yticklabels=[])
    ax1.set_title('True')
    # handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.set(xlabel=' ', ylabel=' ')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.01), ncol=1, title='True labels', title_fontsize=20, fontsize=15)
    # legend1 = g1.legend(loc='upper center', bbox_to_anchor=(0, -0.1), ncol=1, title='True labels',
    #                      title_fontsize=20, fontsize=15)
    # ax1.add_artist(legend1)

    g2 = sns.scatterplot(data=df, x='x', y='y', hue='predLabels', alpha=0.3, s=18, ax=ax2, palette=pallette2)
    g2.set(xticklabels=[], xticks=[], yticks=[], yticklabels=[])
    ax2.set_title('DeepDPM')
    # handles1, labels1 = ax1.get_legend_handles_labels()
    ax2.set(xlabel=' ', ylabel=' ')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, -0.01), ncol=1, title='Predicted labels', title_fontsize=20, fontsize=15)
    # legend2 = g2.legend(loc='upper center', bbox_to_anchor=(0, -0.1), ncol=1, title='True labels',
    #                      title_fontsize=20, fontsize=15)
    # ax2.add_artist(legend2)

    plt.tight_layout()
    # ax1.set_title(f"N={N}, S={S}, {val_train if val_train=='train' else val_train+'idation'} data, {'Flat input' if isFlat else 'Unflat input'}, Channels: {n_channels}")
    # sns.move_legend(ax1, "upper left", bbox_to_anchor=(0, -0.01))
    fig.savefig(f'{path_to_save}/{figFileName}', dpi=300, bbox_inches='tight')
    print(f'saved figure to {path_to_save}/{figFileName}')

    #### write the path to a file
    # with open("/home/labs/testing/class49/All_results/DeepDPM/saved_figures.txt",a) as f:
    #     f.write(f'{path_to_data}/{figFileName}')

def get_runs_from_csv(csv_path, column_name):
    df = pd.read_csv(csv_path)
    runs = df[column_name].values
    full_paths = []
    for run in runs:
        # find the file in the folder that includes all of the following in its name: "combinedResults.npz", "Trans-normalize_", "train"
        file = [f for f in os.listdir(run) if f.endswith("combinedResults.npz") and "Trans-normalize" in f and "train" in f][0]
        full_paths.append(os.path.join(run, file))
    return full_paths
if __name__ == "__main__":
    if len(sys.argv) == 1:
        csv_path = "/home/labs/testing/class49/All_results/DeepDPM/MultiRun (1).csv"
        column_name = "parameters/output_dir"
        full_paths = get_runs_from_csv(csv_path, column_name)
        for file_path in full_paths:
            subprocess.run(f"bsub -R rusage[mem=16000] -n 1 -q new-short \" conda activate Peak ; python /home/labs/testing/class49/DeepDPM_original/make_plots.py "
                           f"{file_path} \"", shell=True)
            # break
    else:
        #file_path = "/home/labs/testing/class49/DeepDPM_original/Communities/N3_S0_3Ch/results_LD10_LinAE/N3_S0_3_LinAE__LD10_Trans-normalize_train_0_combinedResults.npz"
        file_path = sys.argv[1]
        main(file_path)

