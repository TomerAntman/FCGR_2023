#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#


####
# TODO: visualize_embeddings is never called because validation_epoch_end is never called.
#  Also, the function"validation_epoch_end" appears twice (AE_ClusterPipeline.py and clusternetasmodel.py)
####
import sys
sys.path.append("/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/FCGR_2023/DeepDPM")
import os
import torch
import argparse
import pytorch_lightning as pl
#from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger
import numpy as np

from src.AE_ClusterPipeline import AE_ClusterPipeline
from src.datasets import CustomDataset #MNIST, REUTERS,
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel
# added by Tomer
from src.feature_engineering_args import add_model_specific_args as feature_engineering_args
from src.clustering_models.clusternet_modules.utils.plotting_utils import PlotUtils, TSNE_comparison

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from src.utils import cluster_acc, check_args
import datetime

import json
import wandb


# evaluation
from scipy.stats import entropy
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import homogeneity_score

#%%
def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--dir", default="/path/to/dataset/", help="dataset directory")
    parser.add_argument("--output_dir", type=str, help="directory for the saved results")
    parser.add_argument("--dataset", default="custom")

    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)"
    )
    parser.add_argument(
        "--batch_size","--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=0, help="number of pre-train epochs"
    )

    parser.add_argument(
        "--pretrain", action="store_true", help="whether use pre-training"
    )

    parser.add_argument(
        "--pretrain_path", type=str, default="./saved_models/ae_weights/mnist_e2e", help="use pretrained weights"
    )
    parser.add_argument(
        "--use_labels_for_eval",
        action = "store_true",
        help="whether to use labels for evaluation"
    )

    # Model parameters
    parser = AE_ClusterPipeline.add_model_specific_args(parser)
    parser = ClusterNetModel.add_model_specific_args(parser)
    parser = feature_engineering_args(parser)

    # Utility parameters
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="number of jobs to run in parallel"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device for computation (default: cpu)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=400,
        help="how many batches to wait before logging the training status",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="short test run on a few instances of the dataset",
    )

    # Logger parameters
    parser.add_argument(
        "--API_key",
        type=str,
        default="your_API_token",
        help="API key for logger (neptune or wandb)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Experiment name and tag",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="random seed",
    )

    parser.add_argument(
        "--features_dim",
        type=int,
        default=128,
        help="features dim of embedded datasets",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="number of AE epochs",
    )
    parser.add_argument(
        "--number_of_ae_alternations",
        type=int,
        default=3,
        help="The number of DeepDPM AE alternations to perform"
    )
    parser.add_argument(
        "--save_checkpoints", type=bool, default=False
    )
    parser.add_argument(
        "--plot_embeddings",
        action = "store_true",
        help="whether to plot embeddings"
    )
    parser.add_argument(
        "--exp_name", type=str, default="default_exp"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run training without Neptune Logger"
    )
    parser.add_argument(
        "--gpus",
        default=None
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the model"
    )
    parser.add_argument(
        "--save_full_model",
        action="store_true",
        help="Save full model (not just state_dict)"
    )
    parser.add_argument(
        "--number_for_logger",
        type=int,
        default=0,
        help="a number that helps to identify the experiment"
    )
    # parser.add_argument(
    #     "--True_k",
    #     type=int,
    #     default=None,
    #     help="True number of clusters"
    # )

    args = parser.parse_args()
    return args


def load_pretrained(args, model):
    if args.pretrain_path is not None and args.pretrain_path != "None":
        # load ae weights
        state = torch.load(args.pretrain_path)
        new_state = {}
        for key in state.keys():
            if key[:11] == "autoencoder":
                new_state["feature_extractor." + key] = state[key]
            else:
                new_state[key] = state[key]

        model.load_state_dict(new_state)

def train_clusternet_with_alternations():
    """
    :return:  net_pred
    """
    # Parse arguments
    args = parse_args()
    args.n_clusters = args.init_k
    if args.output_dir is None:
        args.outpur_dir = os.path.join(args.dir, 'features')
        # /home/labs/testing/class49/DeepDPM_original/Communities/N_3_0/embeddings/
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed:
        pl.utilities.seed.seed_everything(args.seed)

    # Load data
    # if args.dataset == "mnist":
    #     data = MNIST(args)
    # elif args.dataset == "reuters10k":
    #     data = REUTERS(args, how_many=10000)
    # else:
    data = CustomDataset(args)

    train_loader, val_loader = data.get_loaders()
    args.data_dim = data.data_dim
    args = check_args(args, args.latent_dim)


    tags = ['DeepDPM with alternations']
    # args.tag is a string of tags separated by commas (no spaces)
    tags += args.tag.split(',')
    # remove the API_key from the config:
    config = vars(args).copy()
    del config['API_key']
    if args.offline:
        logger = DummyLogger()
    else:
        os.environ["WANDB_API_KEY"] = args.API_key #c184359965e7380901e00bfbfc7c40608302deef
        logger = WandbLogger(
            project="FCGR",
            name=f"{args.exp_name}_{args.dataset}",
            # kwargs for wandb.init
            config=config#,
            #tags=tags + [args.transform_input_data, args.exp_name, args.dataset, args.number_for_logger]
        )

    device = "cuda" if torch.cuda.is_available() and args.gpus is not None else "cpu"

    # Main body
    model = AE_ClusterPipeline(args=args,logger=logger, input_dim=data.data_dim)
    logger.watch(model, log_freq=50)
    if not args.pretrain:
        load_pretrained(args, model)
    if args.save_checkpoints:
        if not os.path.exists(f'./saved_models/{args.dataset}'):
            os.makedirs(f'./saved_models/{args.dataset}')
        os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}')

    max_epochs = args.epoch * (args.number_of_ae_alternations - 1) + 1

    trainer = pl.Trainer(max_epochs=max_epochs, gpus=args.gpus, num_sanity_val_steps=0, checkpoint_callback=False, logger=logger)
    trainer.fit(model, train_loader, val_loader)

    model.to(device=device)
    DeepDPM = model.clustering.model.cluster_model
    DeepDPM.to(device=device)
    net_pred = []
    # evaluate last model
    for i, dataset in enumerate([data.get_train_data(), data.get_validation_data()]):
        data_ = dataset.data
        pred = DeepDPM(data_.to(device=device).float()).argmax(axis=1).cpu().numpy()
        save_name = f"{args.dataset}_{args.exp_name}_{['train','validation'][i]}_{args.number_for_logger}"
        latent_embeddings = DeepDPM.embedding_codes

        # save embeddings
        #torch.save(latent_embeddings,args.output_dir + f"/{save_name}_features.pt")
        # save them also as npy
        #np.save(args.output_dir + f"/{save_name}_features.npy",latent_embeddings.cpu().numpy())
        # save predictions (current format is numpy array)
        #np.save(args.output_dir + f"/{save_name}_pred.npy", pred)
        name_counter = save_npz_file_for_analysis(args.output_dir, save_name, latent_embeddings.cpu().numpy(), pred, dataset.targets.numpy(), args, tags)
        #logger.log_text(f"{['train','test'][i]}/results_saved", 1)
        net_pred.append(pred)
        if args.use_labels_for_eval:
            # Use the labels to evaluate the model
            labels_ = dataset.targets.numpy()
            acc = np.round(cluster_acc(labels_, pred), 5)
            nmi = np.round(NMI(pred, labels_), 5)
            ari = np.round(ARI(pred, labels_), 5)
            logger.log_metrics({f"{['train','val'][i]}/final_NMI": nmi,
                                f"{['train','val'][i]}/final_ACC": acc,
                                f"{['train','val'][i]}/final_ARI": ari})
            print(f"{['Train','Validation'][i]} evaluation ::", f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(pred))}, true K: {len(np.unique(labels_))}")

            evaluation_dict = evaluation(true_labels=labels_, pred_labels=pred, latent_embeddings=latent_embeddings.cpu().numpy())
            log_metrics_dict = {f"{['train','val'][i]}/{k}": v for k, v in evaluation_dict.items()}
            logger.log_metrics(log_metrics_dict)
            # add nmi, acc, ari , and k_gap to the evaluation dict (one line)
            evaluation_dict.update({"nmi": nmi, "acc": acc, "ari": ari, "final_K": len(np.unique(pred)), "true_K": len(np.unique(labels_))})
            logger.log_metrics({f"scores/{['train','val'][i]}": wandb.Table(columns=evaluation_dict.keys(), data=evaluation_dict.values())})#log_table(key=f"scores/{['train','val'][i]}", columns=evaluation_dict.keys(), data=evaluation_dict.values())

            try:
                name_labels = get_name_labels(data_dir=args.dir, labels = labels_)
                TSNE_comparison(latent_embeddings.cpu().numpy(), name_labels, pred, current_epoch=model.current_epoch, alt_num=model.init_clusternet_num, logger=logger, is_final=True)
            except Exception as e:
                # print the error:
                print("TSNE comparison failed. Error: ", e)

            # if i == 0:
            #     print("Train evaluation:")
            # else:
            #     print("Validation evaluation")
            # print(f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(pred))}")

    # save model
    if args.save_model:
        os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}', exist_ok=True)

        torch.save(model.state_dict(), f'./saved_models/{args.dataset}/{args.exp_name}/AE_ClusterPipeline_{args.number_for_logger}.pth')
        torch.save(DeepDPM.state_dict(), f'./saved_models/{args.dataset}/{args.exp_name}/DeepDPM_{args.number_for_logger}.pth')
        if args.save_full_model:
            try:
                torch.save(DeepDPM, f'./saved_models/{args.dataset}/{args.exp_name}/DeepDPM-full.pth')
            except ReferenceError:
                print("Could not save full model")
    # if you want to load the model from scratch and use it:
        # DeepDPM = AE_ClusterPipeline(args=args, logger=..., input_dim=data.input_dim)
        # DeepDPM.load_state_dict(torch.load(f'./saved_models/{args.dataset}/{args.exp_name}/DeepDPM.pth'))
        # DeepDPM.eval()
        # DeepDPM.freeze()
        # DeepDPM.to(device=device)
        # data = data.get_test_data().to(device=device).float()
        # pred = DeepDPM(data).argmax(axis=1).cpu().numpy()


    model.cpu() # Free up GPU memory
    DeepDPM.cpu() # Free up GPU memory

    # Return the nets predictions for the train and validation sets
    return net_pred

def save_npz_file_for_analysis(output_dir, save_name, latent_embeddings, pred, labels, args, tags):
    # make a dictionary with all the information
    runInfo = {}
    runInfo['args'] = vars(args)
    runInfo['tags'] = tags
    runInfo['time'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # check if file exists and add a number to the name if it does
    npz_name = f"{save_name}_combinedResults"
    save_path = os.path.join(output_dir,npz_name)
    counter=1
    while os.path.exists(save_path+".npz"):
        npz_name = f"{save_name}_combinedResults_{counter}"
        save_path = os.path.join(output_dir,npz_name)
        counter+=1

    np.savez(f"{save_path}.npz", features=latent_embeddings, labels=labels, predLabels=pred)
    with open(f"{save_path}_runInfo.json", "w") as f:
        json.dump(runInfo, f)


    # # Within a folder called "completeRuns", save a dummy file with the complete
    # # This is used to check if a run is complete
    # try:
    #     completeRuns_path = "/home/labs/testing/class49/DeepDPM_original/completeRuns"
    #     os.makedirs(completeRuns_path, exist_ok=True)
    #     with open(os.path.join(completeRuns_path,npz_name), "w") as f:
    #         pass
    # except:
    #     pass

    return counter

def get_name_labels(data_dir, labels):
    # The folder that contained the numpy files of the codes and labels also contains a file called "integer_labels_dict.txt"
    # This file contains the mapping between the integer labels and the string labels
    #labels = self.targets.numpy()
    with open(os.path.join(data_dir, "integer_labels_dict.txt"), "r") as f:
        name_dict = f.read()
    name_dict = eval(name_dict)
    # the keys are the full names and the values are the integer labels
    # switch the keys and values:
    name_dict = {v: k for k, v in name_dict.items()}
    name_labels = [name_dict[label] for label in labels]
    return name_labels

### EVALUATION FUNCTIONS ###
def confmat_scores(confmat):
    order  = confmat.max(axis=0).argsort()[::-1]
    used_clusters = []
    accuracies = []
    for i in order:
        vec = confmat[:,i]
        cluster, maxval = vec.argmax(), vec.max()
        if cluster in used_clusters:
            accuracy = 0
        else:
            accuracy = maxval / vec.sum() # actually "sensitivity"?
            used_clusters.append(cluster)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    real_bal_accuracy = accuracies.mean()
    real_accuracy     = (accuracies * (confmat.sum(axis=0) / confmat.sum())).sum()
    order  = confmat.max(axis=1).argsort()[::-1]
    used_clusters = []
    accuracies = np.zeros(confmat.shape[0])
    for i in order:
        vec = confmat[i]
        cluster, maxval = vec.argmax(), vec.max()
        if cluster in used_clusters:
            accuracy = 0
        else:
            accuracy = maxval / vec.sum()
            used_clusters.append(cluster)
        accuracies[i] = accuracy
    accuracies = np.array(accuracies)
    predicted_bal_accuracy = accuracies.mean()
    predicted_accuracy     = (accuracies * (confmat.sum(axis=1) / confmat.sum())).sum()

    # should I save the confmat in a different CSV?
    return real_accuracy, real_bal_accuracy, predicted_accuracy, predicted_bal_accuracy
def evaluation(true_labels, pred_labels, latent_embeddings):
    """
    :param true_labels: the true labels of the data (numpy array)
    :param pred_labels: the predicted labels of the data (numpy array)
    :param latent_embeddings: the latent embeddings of the data (numpy array shape: (num_samples, num_latent_dims))
    """
    unique_real_clusters      = np.sort(np.unique(true_labels))
    unique_predicted_clusters = np.sort(np.unique(pred_labels))

    confmat = np.zeros((unique_predicted_clusters.size, unique_real_clusters.size), dtype=int)
    for I in range(len(true_labels)):
        i = (unique_predicted_clusters == pred_labels[I]).argmax()
        j = (unique_real_clusters == true_labels[I]).argmax()
        confmat[i,j] += 1
    entropy_real_clusters      = entropy(confmat / confmat.sum(axis=0), axis=0, base=confmat.shape[0])
    entropy_predicted_clusters = entropy(confmat / confmat.sum(axis=1, keepdims=True), axis=1, base=confmat.shape[1])

    mean_entropy_predicted_clusters = entropy_predicted_clusters.mean()
    Wmean_entropy_predicted_clusters = (entropy_predicted_clusters * (confmat.sum(axis=1) / confmat.sum(axis=1).sum())).sum()

    mean_entropy_real_clusters = entropy_real_clusters.mean()
    Wmean_entropy_real_clusters = (entropy_real_clusters * (confmat.sum(axis=0) / confmat.sum(axis=0).sum())).sum()

    # Silhouette score: an array-like of shape (n_samples_a, n_features)  is needed. n_samples_a is the number of samples, and n_features is based on the latent space dimension
    # The silhouette score is calculated for each sample and is based on the mean intra-cluster distance and the mean nearest-cluster distance for each sample
    silhouetteScore  = silhouette_score(latent_embeddings, true_labels)
    # Homogeneity Score:
    homogeneityScore = homogeneity_score(true_labels, pred_labels)

    precision = confmat.max(axis=1).sum() / confmat.sum()
    recall    = confmat.max(axis=0).sum() / confmat.sum()
    F1 = 2 * (precision*recall) / (precision+recall)

    balanced_precision = (confmat.max(axis=1) / confmat.sum(axis=1)).mean()
    balanced_recall    = (confmat.max(axis=0) / confmat.sum(axis=0)).mean()
    balanced_F1        = 2 * (balanced_precision*balanced_recall) / (balanced_precision+balanced_recall)

    real_accuracy, real_bal_accuracy, predicted_accuracy, predicted_bal_accuracy = confmat_scores(confmat)
    new_eval_bal = 1-((1 - real_bal_accuracy)**2 + (1- predicted_bal_accuracy)**2)**0.5
    new_eval = 1-((1 - real_accuracy)**2 + (1- predicted_accuracy)**2)**0.5

    # evaluation_list = [
    #     mean_entropy_predicted_clusters,     Wmean_entropy_predicted_clusters,
    #     mean_entropy_real_clusters,          Wmean_entropy_real_clusters,
    #     silhouetteScore, homogeneityScore,
    #     precision,           recall,             F1,
    #     balanced_precision,  balanced_recall,    balanced_F1,
    #     real_accuracy,       real_bal_accuracy,
    #     predicted_accuracy,  predicted_bal_accuracy
    # ]
    evaluation_dict = {
        "mean_entropy_predicted_clusters": mean_entropy_predicted_clusters, "Wmean_entropy_predicted_clusters": Wmean_entropy_predicted_clusters,
        "mean_entropy_real_clusters": mean_entropy_real_clusters, "Wmean_entropy_real_clusters": Wmean_entropy_real_clusters,
        "silhouetteScore": silhouetteScore, "homogeneityScore": homogeneityScore,
        "precision": precision, "recall": recall, "F1": F1,
        "balanced_precision": balanced_precision, "balanced_recall": balanced_recall, "balanced_F1": balanced_F1,
        "real_accuracy": real_accuracy, "real_bal_accuracy": real_bal_accuracy,
        "predicted_accuracy": predicted_accuracy, "predicted_bal_accuracy": predicted_bal_accuracy,
        "new_eval_bal": new_eval_bal, "new_eval": new_eval
    }

    return evaluation_dict





if __name__ == "__main__":
    train_clusternet_with_alternations()
