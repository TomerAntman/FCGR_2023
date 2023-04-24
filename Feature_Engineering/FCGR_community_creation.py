"""
Given a list of isolates and a flag of whether to make the community balanced or unbalanced,
this script creates a numpy file of FCGRs, composed of the FCGRs of the isolates in the list.
"""
import argparse
import os
import sys
import numpy as np

def get_community_FCGRs(isolate_names, community_folder, ID, args):
    folder = '/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Isolates'
    unbalanced = args.unbalanced
    for i, isolate_name in enumerate(isolate_names):
        if not os.path.exists(os.path.join(folder,isolate_name)):
            raise FileNotFoundError(f"Isolate {isolate_name} does not exist.")
        if unbalanced:
            raise NotImplementedError
        else:
            n_reads = args.n_reads # 1.5K
            # TODO: implement unbalanced community
        FCGR_reads, read_names = get_reads(isolate_name, folder, ID, args, i)
        # 10% of the reads will be used for validation:
        n_validation = int(0.1 * n_reads)
        n_train = n_reads - n_validation

        # the dimensions of the FCGRs are (n_reads, 2**kmer, 2**kmer, channels)
        if len(FCGR_reads.shape) == 3:
            #f"FCGRs should be 4-dimensional (n_reads, 2**kmer, 2**kmer, channels). Got {len(FCGR_reads.shape)} dimensions: {FCGR_reads.shape}"
            FCGR_reads = np.expand_dims(FCGR_reads, axis=3)
        if i == 0:
            FCGRs_train = np.zeros((n_train*len(isolate_names), FCGR_reads.shape[1], FCGR_reads.shape[2], FCGR_reads.shape[3]))
            FCGRs_validation = np.zeros((n_validation*len(isolate_names), FCGR_reads.shape[1], FCGR_reads.shape[2], FCGR_reads.shape[3]))
            read_names_train = np.zeros(n_train*len(isolate_names), dtype=object)
            read_names_validation = np.zeros(n_validation*len(isolate_names), dtype=object)

        FCGRs_train[i * n_train:(i + 1) * n_train,:,:,:] = FCGR_reads[:n_train]
        FCGRs_validation[i * n_validation:(i + 1) * n_validation,:,:,:] = FCGR_reads[n_train:]
        read_names_train[i * n_train:(i + 1) * n_train] = read_names[:n_train]
        read_names_validation[i * n_validation:(i + 1) * n_validation] = read_names[n_train:]

    return FCGRs_train, FCGRs_validation, read_names_train, read_names_validation


def get_reads(isolate_name,parent_folder, FCGR_ID,args, i):
    # read_length_CGR = args.read_length
    # ec_CGR = args.ec
    # CGR_folder = os.path.join(parent_folder,f'{isolate_name}',f'CGR_l{read_length_CGR}_ec{ec_CGR}')
    # kmer, IPD_stride, IPD_r, channels, IPD_stratification, read_length, effective_coverage = \
    #     [args.__dict__[key] for key in ['kmer','IPD_stride','IPD_r','channels','IPD_stratification','read_length','effective_coverage']]
    # FCGR_ID = f"{kmer}_{IPD_stride}_{IPD_r}_{channels}_" \
    #           f"{IPD_stratification}_{read_length}_{round(effective_coverage,1)}"
    # FCGR_folder = os.path.join(CGR_folder,f"FCGR_{FCGR_ID}")

    isolate_folder = os.path.join(parent_folder, isolate_name)
    # os.walk:
    found= False
    for root, dirs, files in os.walk(isolate_folder):
        for dir in dirs:
            if dir == FCGR_ID:
                FCGR_folder = os.path.join(root, dir)
                found = True
                break
    assert found, f"FCGR folder {FCGR_ID} not found in {isolate_folder}"

    # Load the NPZ file of the FCGRs of the isolate (FCGR.npz):
    FCGRs = np.load(os.path.join(FCGR_folder,'FCGR.npz'))["FCGR_arrays"]

    # randomly choose n_reads from the FCGRs (theirs indices):
    seed = args.seed + i
    np.random.seed(seed)
    indices = np.random.choice(FCGRs.shape[0], args.n_reads, replace=False)
    FCGR_reads = FCGRs[indices]

    # load names from FCGR_read_names.txt:
    with open(os.path.join(FCGR_folder,'FCGR_read_names.txt'),'r') as f:
        read_names = f.readlines()
        read_names = [read_name.strip().split("__")[0] for read_name in read_names]
    read_names = np.array(read_names)
    read_names = read_names[indices]
    read_names = np.core.defchararray.add(isolate_name+"__", read_names)

    return FCGR_reads, read_names


def save_npy(FCGRs_train, FCGRs_validation, read_names_train, read_names_validation, full_path, seed):

    # SHUFFLE
    np.random.seed(seed)
    indices_train = np.random.permutation(FCGRs_train.shape[0])
    indices_validation = np.random.permutation(FCGRs_validation.shape[0])

    # TRAIN
    ## save npy of the FCGRs
    np.save(os.path.join(full_path,'train_data.npy'), FCGRs_train[indices_train])

    ## save read names (full names with isolate label)
    with open(os.path.join(full_path,'train_read_names.txt'),'w') as f:
        for read_name in read_names_train[indices_train]:
            f.write(read_name + '\n')

    ## save labels (isolate labels only)
    train_labels = [read_name.split("__")[0] for read_name in read_names_train[indices_train]]
    # create a dictionary to transform the labels to integers:
    integer_labels = np.unique(train_labels)
    integer_labels = {integer_labels[i]:i for i in range(len(integer_labels))}
    # transform the labels to integers:
    train_labels_as_integers = [integer_labels[label] for label in train_labels]
    with open(os.path.join(full_path,'train_labels.txt'),'w') as f:
        for label in train_labels_as_integers:
            f.write(f"{label}\n")

    ## save dictionary of labels
    n_labels = len([*integer_labels])
    counter=1
    with open(os.path.join(full_path,'integer_labels_dict.txt'),'w') as f:
        for key, value in integer_labels.items():
            if counter==1:
                f.write("{\n"+f"'{key}': {value},\n")
            elif counter==n_labels:
                f.write(f"'{key}': {value}"+ "\n}")
            else:
                f.write(f"'{key}': {value},\n")
            counter+=1


    # VALIDATION
    ## save npy of the FCGRs
    np.save(os.path.join(full_path,'validation_data.npy'), FCGRs_validation[indices_validation])

    ## save read names (full names with isolate label)
    with open(os.path.join(full_path,'validation_reads_names.txt'),'w') as f:
        for read_name in read_names_validation[indices_validation]:
            f.write(read_name + '\n')

    ## save labels (isolate labels only)
    validation_labels = [read_name.split("__")[0] for read_name in read_names_validation[indices_validation]]
    validation_labels_as_integers = [integer_labels[label] for label in validation_labels]
    with open(os.path.join(full_path,'validation_labels.txt'),'w') as f:
        for label in validation_labels_as_integers:
            f.write(f"{label}\n")



def get_isolates(community_folder):
    "The community folder contains a text file with the isolate names."
    with open(os.path.join(community_folder,'isolates.txt'),'r') as f:
        isolate_names = f.readlines()
        isolate_names = [isolate_name.strip() for isolate_name in isolate_names]
    return isolate_names

def parse():
    args = argparse.ArgumentParser()
    args.add_argument('--N', type=int, default=8)
    args.add_argument('--S', type=int, default=99)
    args.add_argument('--kmer', type=int, default=5)
    args.add_argument('--IPD_stride', type=str, default="max")
    args.add_argument('--IPD_r', type=float, default=1.0)
    args.add_argument('--channels', type=int, default=3, choices=[1,3])
    args.add_argument('--IPD_stratification', type=str, default="MeanStd")
    args.add_argument('--read_length', type=int, default=10_000)
    args.add_argument('--effective_coverage', type=float, default=10.0)
    args.add_argument('--n_reads', type=int, default=1_500)

    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--unbalanced', action='store_true')

    return args.parse_args()

if __name__ == '__main__':
    """
    Given N and S, the path to the community is clear:
    f'/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Communities/N{N}_S{S}/'
    In this folder, there's a txt file named "isolates.txt" which contains the names of the isolates in the community (one per line).
    
    This folder also contains FCGR folders with the "ID" as their name, which is the following:
    f"FCGR_{kmer}_{IPD_stride}_{IPD_r}_{channels}_{IPD_stratification}_{read_length}_{round(effective_coverage,1)}"
    (If it doesn't exist, it will be created).
    This folder contains 2 subfolders (balanced and unbalanced) which contain the FCGRs of the community.
    (If they don't exist, they will be created).
    
    The ID allows us to know where to look for the FCGRs of the community (os.walk until we find the ID folder for each isolate).
    
    N, S, kmer, IPD_stride, IPD_r, channels, IPD_stratification, read_length, effective_coverage are all given as arguments.
    """
    args = parse()
    N = args.N
    S = args.S
    community_name = f"N{N}_S{S}"
    community_folder = f'/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Communities/{community_name}/'
    ID = f"FCGR_{args.kmer}_{args.IPD_stride}_{args.IPD_r}_{args.channels}_" \
            f"{args.IPD_stratification}_{args.read_length}_{round(args.effective_coverage,1)}"
    isolates = get_isolates(community_folder)
    FCGRs_train, FCGRs_validation, read_names_train, read_names_validation = get_community_FCGRs(isolates, community_folder, ID, args)
    # create the community folder with ID if it doesn't exist:
    community_folder_ID = os.path.join(community_folder,ID)
    os.makedirs(community_folder_ID, exist_ok=True)
    # create the balanced and unbalanced folders if they don't exist:
    full_path = os.path.join(community_folder_ID,'unbalanced') if args.unbalanced else os.path.join(community_folder_ID,'balanced')
    os.makedirs(full_path, exist_ok=True)
    full_path = os.path.join(full_path, "Data")
    os.makedirs(full_path, exist_ok=True)
    save_npy(FCGRs_train, FCGRs_validation, read_names_train, read_names_validation, full_path, args.seed)



