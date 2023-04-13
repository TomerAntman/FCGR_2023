#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers.base import DummyLogger
import numpy as np

from src.AE_ClusterPipeline import AE_ClusterPipeline
from src.datasets import CustomDataset #MNIST, REUTERS,
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from src.utils import cluster_acc, check_args
import datetime

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
        "--tag",
        type=str,
        default="default",
        help="Experiment name and tag",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        help="a number that helps to identify the experiment"
    )
    parser.add_argument(
        "--True_k",
        type=int,
        default=None,
        help="True number of clusters"
    )
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
    if args.offline:
        logger = DummyLogger()
    else:
        logger = NeptuneLogger(
                api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3M2NkOGE0MS1mZGRiLTQ3M2EtOWFlMS0wMmU1MGY3YzVkZjUifQ==',
                project_name='DeepDPM-FCGR/MultiRun',
                experiment_name=args.exp_name,
                #name = f"{args.exp_name}_{args.dataset}",
                params=vars(args),
                tags=tags + [args.transform_input_data, args.exp_name, args.dataset, args.number_for_logger],
            )


    device = "cuda" if torch.cuda.is_available() and args.gpus is not None else "cpu"
    if isinstance(logger, NeptuneLogger):
        if logger.api_key == 'your_API_token':
            print("No Neptune API token defined!")
            print("Please define Neptune API token or run with the --offline argument.")
            print("Running without logging...")
            logger = DummyLogger()

    # Main body
    model = AE_ClusterPipeline(args=args, logger=logger, input_dim=data.data_dim)
    if not args.pretrain:
        load_pretrained(args, model)
    if args.save_checkpoints:
        if not os.path.exists(f'./saved_models/{args.dataset}'):
            os.makedirs(f'./saved_models/{args.dataset}')
        os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}')

    max_epochs = args.epoch * (args.number_of_ae_alternations - 1) + 1

    trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, gpus=args.gpus, num_sanity_val_steps=0, checkpoint_callback=False)
    trainer.fit(model, train_loader, val_loader)

    model.to(device=device)
    DeepDPM = model.clustering.model.cluster_model
    DeepDPM.to(device=device)
    net_pred = []
    # evaluate last model
    for i, dataset in enumerate([data.get_train_data(), data.get_test_data()]):
        data_ = dataset.data
        pred = DeepDPM(data_.to(device=device).float()).argmax(axis=1).cpu().numpy()
        save_name = f"{args.dataset}_{args.exp_name}_{['train','test'][i]}_{args.number_for_logger}"
        latent_embeddings = DeepDPM.embedding_codes

        # save embeddings
        #torch.save(latent_embeddings,args.output_dir + f"/{save_name}_features.pt")
        # save them also as npy
        #np.save(args.output_dir + f"/{save_name}_features.npy",latent_embeddings.cpu().numpy())
        # save predictions (current format is numpy array)
        #np.save(args.output_dir + f"/{save_name}_pred.npy", pred)
        name_counter = save_npz_file_for_analysis(args.output_dir, save_name, latent_embeddings.cpu().numpy(), pred, dataset.targets.numpy(), args, tags)
        logger.log_metric(f"{['train','test'][i]}/results_saved", 1)
        net_pred.append(pred)
        if args.use_labels_for_eval:
            # Use the labels to evaluate the model
            labels_ = dataset.targets.numpy()
            acc = np.round(cluster_acc(labels_, pred), 5)
            nmi = np.round(NMI(pred, labels_), 5)
            ari = np.round(ARI(pred, labels_), 5)
            if i == 0:
                print("Train evaluation:")
            else:
                print("Validation evaluation")
            print(f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(pred))}")
            logger.log_metric(f"{['train','val'][i]}/final_NMI", nmi)
            logger.log_metric(f"{['train','val'][i]}/final_ACC", acc)
            logger.log_metric(f"{['train','val'][i]}/final_ARI", ari)
    # save model
    if args.save_model:
        os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}', exist_ok=True)

        torch.save(model.state_dict(), f'./saved_models/{args.dataset}/{args.exp_name}/AE_ClusterPipeline_{args.number_for_logger}.pth')
        torch.save(DeepDPM.state_dict(), f'./saved_models/{args.dataset}/{args.exp_name}/DeepDPM_{args.number_for_logger}.pth')
        if args.save_full_model:
            torch.save(DeepDPM, f'./saved_models/{args.dataset}/{args.exp_name}/DeepDPM-full.pth')
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

    np.savez(f"{save_path}.npz", features=latent_embeddings, labels=labels, predLabels=pred, runInfo=runInfo)
    # Within a folder called "completeRuns", save a dummy file with the complete
    # This is used to check if a run is complete
    try:
        completeRuns_path = "/home/labs/testing/class49/DeepDPM_original/completeRuns"
        os.makedirs(completeRuns_path, exist_ok=True)
        with open(os.path.join(completeRuns_path,npz_name), "w") as f:
            pass
    except:
        pass

    return counter

if __name__ == "__main__":
    train_clusternet_with_alternations()
