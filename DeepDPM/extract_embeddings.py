print("Loading dependencies...")
import time
t0 = time.time()
import argparse
import torch
import numpy as np
import os
import glob
# from src.datasets import MNIST, CIFAR10, USPS
# from src.embbeded_datasets import embbededDataset
from src.AE_ClusterPipeline import AE_ClusterPipeline
from src.datasets import CustomDataset
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel
print(f"Loading dependencies took {time.time() - t0} seconds")

def parse_minimal_args(parser):
    # Dataset parameters
    parser.add_argument("--dir", default="/path/to/datasets", help="datasets directory")
    parser.add_argument("--output_dir", default="None", type=str, help="save results in this directory")
    parser.add_argument("--dataset", default="mnist", help="the dataset used")

    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "--ae_pretrain_path", type=str, default="/path/to/ae/weights", help="the path to pretrained ae weights"
    )
    parser.add_argument(
        "--umap_dim", type=int, default=10
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--imbalanced", type=bool, default=False
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1
    )
    parser.add_argument(
        "--gpus", type=int, default=0
    )
    parser.add_argument(
        "--device", type=str, default='cuda'
    )

    parser.add_argument(
        "--features_dim", '-fm',
        type=int,
        default=128,
        help="features dim of embedded datasets",
    )
    #parser.add_argument(
    #    "--autoencoder",
    #    type=str,
    #    default="Conv2DAutoEncoder",
    #    help="choose an autoencoder architecture. Options: Autoencoder, ConvAutoencoder, Conv2DAutoEncoder"
    #)

    parser.add_argument(
        "--use_labels_for_eval",
        action = "store_true",
        help="whether to use labels for evaluation"
    )
    return parser

def make_embbedings(args):
    if args.output_dir=="None":
        args.output_dir = f"{args.dir}/results"
        os.makedirs(args.output_dir, exist_ok=True)
    data = CustomDataset(args)

    train_loader, val_loader = data.get_loaders()
    args.data_dim = data.data_dim
    args.input_dim = data.args.input_dim
    #check_args(args, args.latent_dim)

#    train_loader, val_loader = data.get_loaders(args)
#    args.input_dim = data.data_dim

    # Main body
    ae = AE_ClusterPipeline(args=args, logger=None, input_dim=args.data_dim)
    ae.load_state_dict(torch.load(args.ae_pretrain_path))
    ae.eval()
    ae.freeze()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus is not None else "cpu")
    ae.to(device)

    train_codes, train_labels = [], []
    val_codes, val_labels = [], []

    for i, data in enumerate(train_loader, 0):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].to(device)
            # if args.dataset == "mnist":
            #     inputs = inputs.view(inputs.size()[0], -1)
            codes = torch.from_numpy(ae.forward(inputs, latent=True)) # get latents
            train_codes.append(codes.view(codes.shape[0], -1)) # append batches
            train_labels.append(labels)

    train_codes = torch.cat(train_codes).cpu().numpy()
    train_labels = torch.cat(train_labels).cpu().numpy()
    return train_codes, train_labels


def run_make_embbedings(model_path):
    parser = argparse.ArgumentParser(description="Only_for_embbedding")
    parser = parse_minimal_args(parser)
    parser = AE_ClusterPipeline.add_model_specific_args(parser)
    parser = ClusterNetModel.add_model_specific_args(parser)
    args = parser.parse_args()
    print('-' * 50)
    args.ae_pretrain_path = model_path
    path_parts = args.ae_pretrain_path.split("/")
    args.output_dir = os.path.join("/", *path_parts[:-1])
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.output_dir, "pretrained_train_embeddings.npz")):
        print("Embeddings already exists for: ", model_path)
    else:
        print("Collecting embeddings for {} model".format(args.ae_pretrain_path))
        N = path_parts[7].split("_")[0][1:]
        S = path_parts[7].split("_")[1][1:]
        ch = path_parts[8].split("_")[0]
        latent_dim = path_parts[8].split("__LD")[1].split("_")[0]
        args.dir = f"/home/labs/testing/class49/DeepDPM_original/Communities/N{N}_S{S}_{ch}Ch"
        args.latent_dim = int(latent_dim)
        autoencoder = path_parts[8].split("_")[1]
        if autoencoder == "ConvAE":
            args.ConvVAE = True
        else:
            args.ConvVAE = False
        train_codes, train_labels = make_embbedings(args)
        print(f"Saving embeddings to {args.output_dir}")
        np.savez(os.path.join(args.output_dir, "pretrained_train_embeddings.npz"), features=train_codes, labels=train_labels)


if __name__ == "__main__":
    dir = "/home/labs/testing/class49/DeepDPM_original/saved_models/"#N6_S0/1_ConvAE__LD8_Trans-normalize/trained_model_350Epochs"
    dirs_to_run = [dir]
    failed = []
    for i in range(3):
        if len(dirs_to_run)==0:
            print("Sleeping for 1 minute...")
            time.sleep(300)
        while len(dirs_to_run)>0:
            current_dir = dirs_to_run.pop()
            dirs_to_run.extend([os.path.join(current_dir,subdir) for subdir in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir,subdir))])
            for model_path in glob.glob(os.path.join(current_dir, "*Epochs")):
                run_make_embbedings(model_path)
    print("Done")