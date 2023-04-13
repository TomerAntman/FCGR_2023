import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import pandas as pd
import numpy as np
import seaborn as sns
import time
import os
# scores:
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import argparse
from sklearn.metrics import fowlkes_mallows_score as fms
from sklearn.cluster import DBSCAN

#%%
# DOCUMENTATION:
# Given N,S, and n_channels, a .npz file is loaded and it includes the data and the labels. use it to get the full names of the true labels.

# data = dict(np.load(f"/home/labs/testing/class49/PreProcess/communities/N_{N}/FCGR_files/community_S-{S}_N-{N}_{n_channels}Ch.npz"))

# Given a features_path, the information can be exctracted based on the name of the file.
# run_info = extract_run_info(features_path) # /home/labs/testing/class49/DeepDPM_original/Communities/N3_S0_3/results/N_3_0__train_latentWithPredLabels.npz

def extract_run_info(features_path):
    """
    Extracts the run information from the features_path.
    :param features_path:
    :return:
    """
    pass


def make_df_from_embeddings(true_labels, pred_labels, mapper,data,DBSCAN_pred):
    df = pd.DataFrame({'x': mapper[:, 0], 'y': mapper[:, 1], 'TrueNumLabel': true_labels, 'fullLabel':' ', 'PredLabel':pred_labels, 'DBSCAN_labels':DBSCAN_pred})
    #load full names of labels
    unique_num_labels = np.unique(true_labels)
    full_name_vector = np.repeat(np.nan,len(true_labels)).astype(str)
    for label in unique_num_labels:
        idx = np.where(data['number_labels'] == label)[0][0]
        full_name = data['labels'][idx]
        full_name_vector[df.TrueNumLabel == label] = full_name
        # df["fullLabel"][df.TrueNumLabel == label] = full_name
    df['fullLabel'] = full_name_vector
    df = df.sort_values(by=['fullLabel'])
    return df



##### add the full name of the labels to the dataframe!
# def add_full_labels_to_df(df):
#     data = dict(np.load(f"/home/labs/testing/class49/PreProcess/communities/N_{N}/FCGR_files/community_S-{S}N-{N}{n_channels}Ch.npz"))
#     unique_num_labels = np.unique(labels)
#     for label in unique_num_labels:
#         idx = np.where(data['number_labels']==label)[0][0]
#         full_name = data['labels'][idx]
#         df['fullLabel'][df.numLabel==label] = full_name
#     print('collected labels')
#     df = df.sort_values(by=['fullLabel'])
#     return df

def DBSCAN_clustering(X, hypers_path):
    # load hyper parameters
    hypers_pickle = np.load(hypers_path, allow_pickle=True)
    reduction_method = hypers_pickle['reduction_method']
    reduction_params = hypers_pickle['reduction_params']
    clustering_params =  hypers_pickle['clustering_params']
    if reduction_method == 'pca':
        mapper = PCA(n_components=reduction_params['n_components'])
        reduced_X = mapper.fit_transform(X)
    elif reduction_method == 'tsne':
        mapper = TSNE(n_components=reduction_params['n_components'])
        reduced_X = mapper.fit_transform(X)
    elif reduction_method == 'umap':
        mapper = umap.UMAP(n_components=reduction_params['n_components'])
        reduced_X = mapper.fit_transform(X)
    else:
        raise ValueError("reduction method not supported")

    # clustering
    db = DBSCAN(eps=clustering_params['eps'],
                min_samples=clustering_params['min_samples'],
                metric=clustering_params['metric'],
                algorithm=clustering_params['algorithm'],
                leaf_size=clustering_params['leaf_size'])
    y_pred = db.fit_predict(reduced_X, y)
    return y_pred



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



def make_plot(X, y, pred_labels, args, DBSCAN_pred, figFileName = f"Reduced_dimensions_DeepDPM_clustering_plots.png"):
    print("Creating UMAP,tSNE and PCA plots...")
    #S = 0
    #N = 10
    #n_channels = 3


    data = dict(np.load(
        f"/home/labs/testing/class49/PreProcess/communities/N_{args.N}/FCGR_files/community_S-{args.S}_N-{args.N}_{args.n_channels}Ch.npz"))
    pca = PCA(n_components=2)
    pca_mapper = pca.fit_transform(X)
    df_pca = make_df_from_embeddings(y, pred_labels, pca_mapper,data,DBSCAN_pred)
    umap_mapper = umap.UMAP().fit_transform(X)
    df_umap = make_df_from_embeddings(y, pred_labels, umap_mapper,data,DBSCAN_pred)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, metric='cosine')
    tsne_mapper = tsne.fit_transform(X)
    df_tsne = make_df_from_embeddings(y, pred_labels, tsne_mapper,data,DBSCAN_pred)
    # save df to csv
    # df_umap.to_csv(f"{args.features_path[:-4]}_UMAP.csv")
    # df_tsne.to_csv(f"{args.features_path[:-4]}_tSNE.csv")
    # df_pca.to_csv(f"{args.features_path[:-4]}_PCA.csv")

    #load df from csv
    # df_umap = pd.read_csv(f"{args.features_path[:-4]}_UMAP.csv")
    # df_tsne = pd.read_csv(f"{args.features_path[:-4]}_tSNE.csv")
    # df_pca = pd.read_csv(f"{args.features_path[:-4]}_PCA.csv")

    # 2 rows (true labels, predicted labels) and 3 columns (UMAP, tSNE, PCA)
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    #axs[:,:].set(xticklabels=[], xticks=[], yticks=[], yticklabels=[])
    for i,emb in enumerate([('UMAP',df_umap), ('tSNE',df_tsne), ('PCA',df_pca)]):
        name, df = emb
        # scatter plots, using seaborn scatterplot
        g1 = sns.scatterplot(data=df, x='x', y='y', hue='fullLabel', alpha=0.4, s=18, ax=axs[0, i], palette=get_colors(len(df['fullLabel'].unique()), rev=False))
        g1.set(xticklabels=[],xticks=[],yticks=[],yticklabels=[])
        g2 = sns.scatterplot(data=df, x='x', y='y', hue='PredLabel', alpha=0.4, s=18, ax=axs[1, i], palette=get_colors(len(df['PredLabel'].unique()), rev=True))
        g2.set(xticklabels=[],xticks=[],yticks=[],yticklabels=[])
        g3 = sns.scatterplot(data=df, x='x', y='y', hue='DBSCAN_labels', alpha=0.4, s=18, ax=axs[2, i], palette=get_colors(len(df['DBSCAN_labels'].unique()), rev=True))
        g3.set(xticklabels=[],xticks=[],yticks=[],yticklabels=[])
        if i == 0: # the y label is only on the left plot, on the left side of it. And should say "True labels" (or "Predicted labels")
            axs[0,i].set(xlabel=' ', ylabel='True labels')
            axs[1,i].set(xlabel=' ',ylabel='Predicted labels')
            axs[2,i].set(xlabel=' ',ylabel='DBSCAN labels')
        else:
            axs[0,i].set(xlabel=' ', ylabel=' ')
            axs[1, i].set(xlabel=' ', ylabel=' ')
            axs[2, i].set(xlabel=' ', ylabel=' ')
        if i==2:
            axs[0, i].legend(loc='lower left', bbox_to_anchor=(1, 0), ncol=1)
            axs[1, i].legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
            axs[2, i].legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        else:
            axs[0, i].get_legend().remove()
            axs[1, i].get_legend().remove()
            axs[2, i].get_legend().remove()

        axs[0,i].set_title(f"{name}") # only the top plots should have a title (not pred)
    # figure title:
    fig.suptitle(args.run_info, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9) # make room for the title
    print("Done creating UMAP,tSNE and PCA plots")
    #check if output file exists, if so add a number to the end of the file name
    if os.path.exists(os.path.join(args.output_path, figFileName)):
        i = 1
        while os.path.exists(os.path.join(args.output_path, figFileName)):
            figFileName = f"UMAP_tSNE_PCA_S-{args.S}_N-{args.N}_{args.n_channels}Ch_{i}3.png"
            i += 1
    print("Saving plots to: ", f'{args.output_path}/{figFileName}')
    fig.savefig(f'{args.output_path}/{figFileName}', dpi=300, bbox_inches='tight')
    #return ax_true, ax_pred

def DeepDPM_stats(X, y, pred_labels, args,DBSCAN_pred):


    print('-' * 50)
    print(f"Calculation clustering scores for DeepDPM")
    print("Output path: ", args.output_path)
    print('-' * 50)

    t0 = time.time()
    silhouette_score_score = silhouette_score(X, pred_labels)
    print(f"DeepDPM clustering silhouette_score: {silhouette_score_score}\n time: {time.time() - t0}")
    t0 = time.time()
    calinski_harabasz_score_score = calinski_harabasz_score(X, pred_labels)
    # calinski_harabasz_score_score = 0
    print(f"DeepDPM clustering calinski_harabasz_score: {calinski_harabasz_score_score}\n time: {time.time() - t0}")
    t0 = time.time()
    davies_bouldin_score_score = davies_bouldin_score(X, pred_labels)
    # davies_bouldin_score_score = 0
    print(f"DeepDPM clustering davies_bouldin_score: {davies_bouldin_score_score}\n time: {time.time() - t0}")
    linear_assignment_score = fms(y, pred_labels)
    print(f"DeepDPM clustering linear_assignment_score: {linear_assignment_score}")
    n_pred_clusters = len(np.unique(pred_labels))
    print(f"DeepDPM number of clusters: {n_pred_clusters}")
    DBSCNA_linear_assignment_score = fms(y, DBSCAN_pred)
    print(f"DBSCAN clustering linear_assignment_score: {DBSCNA_linear_assignment_score}")
    n_pred_clusters_DBSCAN = len(np.unique(DBSCAN_pred))
    print("DBSCAN number of clusters: ", len(np.unique(DBSCAN_pred)))
    # save stats file with predicted labels scores, and number of clusters
    with open('stats.txt',"wb") as file:
        file.write(f"DeepDPM number of clusters: {n_pred_clusters}\n".encode())
        file.write(f"DeepDPM clustering silhouette_score: {silhouette_score_score}\n".encode())
        file.write(f"DeepDPM clustering calinski_harabasz_score: {calinski_harabasz_score_score}\n".encode())
        file.write(f"DeepDPM clustering davies_bouldin_score: {davies_bouldin_score_score}\n".encode())
        file.write(f"DeepDPM clustering linear_assignment_score: {linear_assignment_score}\n".encode())
        file.write(f"DBSCAN number of clusters: {n_pred_clusters_DBSCAN}\n".encode())
        file.write(f"DBSCAN clustering linear_assignment_score: {DBSCNA_linear_assignment_score}\n".encode())
    print(f'DeepDPM clustering scores saved to {args.output_path}/stats.txt')
    print('-' * 50)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path','-f', type=str, required=True, help='path to npz file')
    parser.add_argument('--pred_labels_path','-p', type=str, required=True, help='path to npz file')
    parser.add_argument('--output_path','-o', type=str, default=None, help='path to output folder')
    parser.add_argument('--indices_path','-i', type=str, default=None, help='path to indices file')
    parser.add_argument('--labels_path','-l', type=str, default=None, help='path to labels file')
    parser.add_argument('--run_info','-info', type=str, default='labels on embeddings', help='name of dataset')
    parser.add_argument('--n_clusters','-c', type=int, default=10, help='number of clusters')
    parser.add_argument('--hypers_path','-hy', type=str, default=None, help='path to hypers file')
    parser.add_argument('--n_channels','-ch', type=int, default=1, help='number of channels')
    parser.add_argument('--S','-S', type=int, default=15, help='number of samples for UMAP')
    parser.add_argument('--N','-N', type=int, default=15, help='number of neighbors for UMAP')
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = os.path.join(*args.features_path.split("/")[:-1])
    args.run_info = args.run_info.replace(" ","_")
    args.output_path = os.path.join(args.output_path, args.run_info+f"_plots")
    os.makedirs(args.output_path, exist_ok=True)
    return args


if __name__=="__main__":
    args = argparser()
    X = np.load(args.features_path)
    pred_labels = np.load(args.pred_labels_path)
    # idx = np.load(args.indices_path)
    # y = np.load(args.labels_path)[idx]
    y = np.load(args.labels_path)['train']
    DBSCAN_pred = DBSCAN_clustering(X, args.hypers_path)
    DeepDPM_stats(X, y, pred_labels, args,DBSCAN_pred)
    make_plot(X, y, pred_labels, args,DBSCAN_pred)
    print("-"*25+"Done"+"-"*25)
    # from torch import load as t_load
    # pt_file = t_load("/home/labs/testing/class49/DeepDPM_original/Communities/N10_S0_3Ch/results_LD6_LinAE/N10_S0_train_features.pt")