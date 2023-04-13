#%% Documentation
"""
This script takes the npy arrays created by "Bam2CGR.py" and creates the FCGRs.

- in __isolate__ mode: The home path needs to be declared, as well as the isolate name.
- in __community__ mode: The home path needs to be declared, as well as the number of isolates in the community (N), and the seed number of the community (S).

There will be an output folder for the isolate/community:
####
/
|--isolate/
| |--CGR_%ID1%/
|   |--CGR_config.yaml
|   |--files/
|   | |--LR_%bamID_1%.npy
|   | |--LR_%bamID_N%.npy
|   |--FCGR_%ID1%/
|   | |--FCGR_config.yaml
|   | | |--files/
|   | | | |--LR_%bamID_1%.npy
|   | | | |--LR_%bamID_N%.npy
####
the ID(1/2/...) is some unique identifier for the configuration of the CGR/FCGR files.
The FCGR_config.yaml file contains the parameters used to create the FCGR files.
The CGR arrays have 3 columns: X, Y, IPDs (the IPDs are processed and normalized but are before the "stride" step).


# Running the code:
1) activate the environment: 'source /home/labs/zeevid/tomerant/FCGR_project/pip_envs/preprocess_env/bin/activate'
2) Activate modules: module load Pysam ; module load Biopython
3) Edit the config.yaml file to your needs (the values will be taken from there if not stated when calling the function)


# Under the hood:
1) read the config.yaml file and parse the arguments.
2) create the output folder if it doesn't exist (save an edited version of the config file there, based on the arguments that were given).
3) create the FCGR files for the isolate or community (if they don't exist).
4) Save the FCGR files in the output folder.
5) plot one of the FCGR files (if the plot flag is on) to see that it looks ok.


# important parameters for FCGR:
- kmer_size: the size of the kmer to use for the FCGR
- IPD_stratification: the method to use for the splitting the Z axis into groups (MeanStd or FullRange)
- IPD_stride: the method to use for the IPD stride (normal, mean, max, emph_A)
- channels: the number of channels to use for the FCGR [1 or 3] (or more, but only implemented for full_range)
"""

#%% Imports
import os
import numpy as np
import yaml
#from Feature_Engineering.FCGR_parser import config_parsing, Parser
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm # for progress bar


#%% The crucial functions
class IPDstride:
    def __init__(self, CGR_array, IPD_stride,  IPD_r, kmer_size):
        """
        :param CGR_array: a numpy array with 4 columns: x, y, IPD, base
        :param IPD_stride: a string, describing the stride method
            - normal: just perform the normal IPD stride (step)
            - mean: The IPD of the kmer is the mean of the IPDs of the bases in the kmer and then do the step with the new IPD values.
            - max: The IPD of the kmer is the max of the IPDs of the bases in the kmer and then do the step with the new IPD values.
            - emph_A: The IPD of the kmer is the mean of the IPDs of the bases in the kmer, but the IPDs of the A bases are emphasized by a factor of 2 and then do the step with the new IPD values.
        :param kmer_size: the size of the kmer to use for the mean/max/emph_A stride types
        """
        self.r = IPD_r
        self.IPDs = CGR_array[:,2]
        self.stride_type = IPD_stride
        self.kmer_size = kmer_size
        self.bases = CGR_array[:,3]
        self.X, self.Y = CGR_array[:,0], CGR_array[:,1]
        # if normal, just do the normal IPD stride:
        if self.stride_type == "normal":
            self.IPD_coords = self.stride()
        else:
            self.trans_function = {"mean": self.by_mean, "max": self.by_max, "emph_A": self.by_mean_emph_A}
            self.IPDs = self.trans_function[self.stride_type]()
            # check the new length of the IPDs array and cut the X and Y arrays accordingly:
            self.IPD_coords = self.stride()


    def IPDstep(self, current_z, i):
        step = (self.IPDs[i] - current_z) * self.r
        new_z = current_z + step
        return new_z

    def stride(self):
        current_z = 0
        IPD_coords = []
        for i in range(len(self.IPDs)):
            new_z = self.IPDstep(current_z, i)
            IPD_coords.append(new_z)
            current_z = new_z
        return IPD_coords

    def by_mean(self):
        trans_IPDs = []
        for i in range(self.kmer_size, len(self.IPDs)+1):
            mean_val = np.mean(self.IPDs[i-self.kmer_size:i])
            trans_IPDs.append(mean_val)
        return trans_IPDs

    def by_max(self):
        trans_IPDs = []
        for i in range(self.kmer_size, len(self.IPDs)+1):
            max_val = np.max(self.IPDs[i-self.kmer_size:i])
            trans_IPDs.append(max_val)
        return trans_IPDs

    def by_mean_emph_A(self):
        trans_IPDs = []
        A_ord = ord('A') # 65
        for i in range(self.kmer_size, len(self.IPDs)+1):
            current_seq_IPDs = self.IPDs[i-self.kmer_size:i]
            current_seq_bases = self.bases[i-self.kmer_size:i]
            # find the indexes of the A bases:
            A_indexes = [j for j, x in enumerate(current_seq_bases) if x == A_ord]
            # emphasize the IPD of the A bases:
            for index in A_indexes:
                current_seq_IPDs[index] = current_seq_IPDs[index]*2
            trans_IPDs.append(np.mean(current_seq_IPDs))
        return trans_IPDs


def CGR2FCGR(CGR_array, kmer_size,IPD_stratification,IPD_stride, IPD_r, channels = 3):
    xy_nbins = 2**kmer_size
    xy_range = (-1,1)
    if channels==1:
        # start from the kmer_size index, because the first kmer_size bases don't have a full kmer to calculate the FCGR
        FCGR_array,_,_ = np.histogram2d(CGR_array[kmer_size:,0], CGR_array[kmer_size:,1], bins=(xy_nbins, xy_nbins),range=((-1,1),(-1,1)))

    else:
        # get the IPD coordinates:
        IPDs = IPDstride(CGR_array, IPD_stride, IPD_r, kmer_size)
        IPDs = IPDs.IPD_coords
        # start from the kmer_size index, because the first kmer_size bases don't have a full kmer to calculate the FCGR
        init_index = 0 if IPD_stride == "normal" else kmer_size-1
        X, Y = CGR_array[init_index:,0], CGR_array[init_index:,1]
        # create the array for the histogramdd function:
        CGR_array = np.column_stack((X,Y,IPDs))

        # get the absolute max coordinate of the IPDs:
        absmax = np.abs(IPDs).max()

        # create the FCGR array (depending on the IPD_stratification):
        if IPD_stratification == "MeanStd":
            z_mean = np.mean(IPDs)
            z_std = np.std(IPDs)
            z_edges = [-absmax, z_mean-z_std, z_mean+z_std, absmax]
            FCGR_array, edges = np.histogramdd(CGR_array, bins=(xy_nbins, xy_nbins, z_edges),range=(xy_range,xy_range,None))
        else: # elif IPD_stratification == "FullRange":
            z_range = [-absmax, absmax]
            FCGR_array, edges = np.histogramdd(CGR_array, bins=(xy_nbins, xy_nbins, channels), range=(xy_range, xy_range, z_range))

    return FCGR_array

def FCGR_normalization(FCGR_array, normalization_method = 'sum_max'):
    # FCGR_normalization: "by_sum" or "by_max" or "by_sum_and_max"
    # by_sum: divide by the sum of all bins to get the frequency (eliminating the effect the length of the sequence)
    # by_max: divide by the maximum value of the FCGR array, stretching the values between 0 and 1
    if "sum" in normalization_method:
        FCGR_array_normed = FCGR_array / FCGR_array.sum()
        if "max" in normalization_method:
            FCGR_array_normed = (FCGR_array_normed - np.min(FCGR_array_normed)) / (np.max(FCGR_array_normed) - np.min(FCGR_array_normed))
    elif "max" in normalization_method:
        FCGR_array_normed = (FCGR_array - np.min(FCGR_array)) / (np.max(FCGR_array) - np.min(FCGR_array))
    elif normalization_method == "l2_norm":
        #FCGR_array_normed = np.linalg.norm(FCGR_array, ord = None, axis = None, keepdims = True)
        FCGR_array_normed = FCGR_array/np.sqrt(np.sum(FCGR_array**2))
    else:
        FCGR_array_normed = FCGR_array

    return FCGR_array_normed

def plot_FCGR(FCGR_array, FCGR_folder, read_name):
    os.makedirs(os.path.join(FCGR_folder, 'images'), exist_ok=True)
    save_path = os.path.join(FCGR_folder, 'images', f"{read_name}.png")
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(FCGR_array, vmin=0, vmax=1, aspect='auto')
    fig.savefig(save_path, dpi=200)


#%% Get CGRs and Filter the reads
def get_CGR(CGR_folder):
    """
    If there's an NPZ file in the CGR folder, load it.
    Else, check if there are NPY files in the CGR folder.
    If there are, load them and concatenate them.
    Else, raise an error.
    Also, if there's an NPZ file but no txt file with the read names, raise an error because it means the NPZ might be incomplete.
    """
    # the NPZ file has the following keys: 'CGR_array', 'read_lengths', 'ecs'
    # If there's an NPZ file, there is also a txt file with the read names (ends with _read_names.txt)
    NPZ_file = [file for file in os.listdir(CGR_folder) if file.endswith(".npz")]

    if len(NPZ_file) == 1 and "CGR_array_read_names.txt" in os.listdir(CGR_folder):
        loaded_npz = np.load(os.path.join(CGR_folder, NPZ_file[0]), allow_pickle=True)
        CGR_arrays, read_lengths, ecs = [loaded_npz[key] for key in loaded_npz.keys()]
        names_file = [file for file in os.listdir(CGR_folder) if file.endswith("CGR_array_read_names.txt")]
        if len(names_file) == 1:
            with open(os.path.join(CGR_folder, names_file[0]), "r") as f:
                read_names = f.read().splitlines()

    else:
        NPY_files = [file for file in os.listdir(CGR_folder) if file.endswith(".npy")]
        if len(NPY_files) == 0:
            raise ValueError(f"There are no CGR files in {CGR_folder}")

        CGR_arrays = np.load(os.path.join(CGR_folder, NPY_files[0]))
        # the names of the files is f"{name}__LEN{actual_read_length}_EC{ec}.npy"
        read_lengths = [int(file.split("__")[-1].split("_")[0][3:]) for file in NPY_files]
        ecs = [int(file.split("__")[-1].split("_")[1][2:].split(".")[0]) for file in NPY_files]
        read_names = [file.split("__LEN")[0] for file in NPY_files]
        for file in NPY_files[1:]:
            CGR_arrays = np.concatenate((CGR_arrays, np.load(os.path.join(CGR_folder, file))))

    return CGR_arrays, read_lengths, ecs, read_names

def filter_CGR(CGR_arrays, read_lengths, ecs, read_names, tags_filter, read_length_threshold):
    """
    :param CGR_arrays: the CGR arrays for all the reads (np.array). 4 columns: x, y, IPDs, base (utf-8, not str)
    :param read_lengths: list of ints. the lengths of the reads
    :param ecs: list. effective coverages of the reads
    :param read_names: names of the reads (strings)
    :param tags_filter: dictionary with the tags to filter the reads
    :param read_length_threshold:  new read length threshold
    :return: filtered CGR arrays, read lengths, effective coverages, read names
    """
    # get the indices of the arrays where the read length is above the threshold and the EC is above tags_filter['ec']:
    indices = np.where((read_lengths >= read_length_threshold) & (ecs >= tags_filter['ec']))[0]
    # filter the arrays:
    filtered_CGR_arrays, filtered_read_lengths, filtered_ecs, filtered_read_names = [], [], [], []
    for i in indices:
        filtered_CGR_arrays.append(CGR_arrays[i])
        filtered_read_lengths.append(read_lengths[i])
        filtered_ecs.append(ecs[i])
        filtered_read_names.append(read_names[i])
    # CGR_arrays = [CGR_arrays[i] for i in indices]
    # read_lengths = [read_lengths[i] for i in indices]
    # ecs = [ecs[i] for i in indices]
    # read_names = [read_names[i] for i in indices]
    return filtered_CGR_arrays, filtered_read_lengths, filtered_ecs, filtered_read_names

#%% Main function
def main(CGR_arrays, read_names, read_lengths, ecs,
         FCGR_folder, IPD_stride, IPD_r, channels, IPD_stratification,
         save_as_npy, save_as_npz, plot_examples):
    """
    :param CGR_arrays: the CGR arrays for all the reads (np.array). 4 columns: x, y, IPDs, base (utf-8, not str)
    :param FCGR_folder: the folder where the FCGR arrays will be saved (string)
    :param IPD_stride: the stride for the IPD (int)
    :param IPD_r: the r for the IPD (int)
    :param channels: the number of channels for the IPD (int)
    :param IPD_stratification: the method for the FCGR (string)
    :param save_as_npy: whether to save the FCGR arrays as npy files (bool)
    :param save_as_npz: whether to save the FCGR arrays as npz files (bool)
    :return: saves the FCGR arrays as npy or npz files
    """
    if plot_examples:
        # make some score to decide which reads to plot
        normed_lengths = np.array(read_lengths)/np.mean(read_lengths)
        normed_ecs = np.array(ecs)/np.mean(ecs)
        read_score = normed_lengths * normed_ecs
        best_ind = np.argmax(read_score)
        worst_ind = np.argmin(read_score)

    # create the FCGR arrays:

    FCGR_arrays = []
    for i, CGR_array in enumerate(tqdm(CGR_arrays)):
        FCGR_array = CGR2FCGR(CGR_array, kmer_size, IPD_stratification, IPD_stride, IPD_r, channels)
        if save_as_npy:
            np.save(os.path.join(FCGR_folder, f"{read_names[i]}.npy"), FCGR_array)
        if save_as_npz:
            FCGR_arrays.append(FCGR_array)

        if plot_examples and (i == best_ind or i == worst_ind):
            for norm_method in ['sum_max']:#, 'max', 'sum', 'l2_norm']:
                normed_array = FCGR_normalization(FCGR_array, normalization_method = norm_method)
                name_for_save = f"{read_names[i]}_{norm_method}"
                plot_FCGR(normed_array, FCGR_folder, read_name=name_for_save)

    # save the FCGR arrays:
    if save_as_npz:
        np.savez(os.path.join(FCGR_folder, "FCGR.npz"), FCGR_arrays=FCGR_arrays)
        with open(os.path.join(FCGR_folder, f"FCGR_read_names.txt"), "w") as fp:
            fp.write('\n'.join(read_names))





#%% Parse the arguments and crearw files/folders
def Parser(params, get_config=False):
    parser = argparse.ArgumentParser(description='Create FCGR arrays from CGR arrays')
    # start by parsing the config path and the mode (isolate or community)
    parser.add_argument("--config_file", "--config", "--config_path",
                        type=str,
                        dest="config_file",
                        default="./config.yaml",
                        help="path to config file"
                        )
    # if you only want the config path, return it
    if get_config:
        simple_args, unknown = parser.parse_known_args()
        config_path = simple_args.config_file
        return config_path

    parser.add_argument("--isolate_or_community", "--icm", #icm stands for "Isolate or Community Mode"
                        type=str,
                        #options=["isolate", "community"],
                        default=params['mode'],
                        help="Isolate or Community Mode. Default is 'isolate'."
                        )
    # else, get these two arguments and continue parsing
    simple_args, unknown = parser.parse_known_args()
    mode = simple_args.isolate_or_community

### Start off with "path parsing" to get the relevant folders of CGR
    if mode == "isolate":
        # isolate name, one of the folder in the isolates_folder
        parser.add_argument("--isolate_name", "--isolate", "--desired_isolate","-i",
                            type=str,
                            default=params['isolate_paths']['isolate_name'],
                            help="Either the name of the isolate, or 'all'. Must be provided if the mode is 'isolate'."
                            )
        # output_folder: default is '/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR' (was '/home/labs/zeevid/Analyses/2022-HGR/HybridIsolates/IPD_CGR')
        parser.add_argument("--output_folder", "--output_path",
                            type=str,
                            default=params['isolate_paths']['output'],
                            help="The folder path in which the output folder of the isolate will be saved. Default is in config.yaml file"
                            )
    elif mode == "community":
        # TODO: add community mode
        raise NotImplementedError("Community mode is not implemented yet.")
    else:
        raise ValueError(f"mode must be 'isolate' or 'community', not {mode}")

### Hyper-parameters parsing:
    ## Of the CGR arrays:
    parser.add_argument("--read_length",
                        type=int,
                        default=params['filters']['read_length'],
                        help="minimal read length to include such as 10000. Default is in config.yaml file"
                        )
    parser.add_argument("--consider_tags",
                        type=str,
                        default= params['filters']['tags']['consider_tags'], #'ec,np,fn,rq', # effective coverage, number of full-length subreads, forward number of complete passes, Predicted average read quality
                        help="Pacbio tags to filter by. Default is 'ec' (effective coverage). Can also include 'np,fn,rq' (number of full-length subreads, forward number of complete passes, Predicted average read quality)"
                        )
    parser.add_argument("--tags_filter", "--tags_threshold",
                        type=str,
                        default=params['filters']['tags']['tags_filter'],#'10,5,2,0.999',
                        help="Thresholds for the tags. Default is '10' (ec=10). Can also be something like '10,5,2,0.999' (ec=10, np=5, fn=2, rq=0.999). Needs to match the tags given in the 'consider_tags' argument"
                        )

    ### New hyper-parameters for FCGR: (CGR hyper-parameters are relevant to find the folders of CGR)
    FCGR_params = params['FCGR'] # ! IMPORTANT

    parser.add_argument("--kmer_size", "-k", "-K",
                        type=int,
                        default=FCGR_params['hyperparams']['kmer_size'],
                        help="kmer size such as 5. Default is in config.yaml file"
                        )
    parser.add_argument("--channels","--multichannel", "--ndim",
                    type=int,
                    default=FCGR_params['hyperparams']['channels'],
                    help="number of channels, must be greater or equal to 1. Default is in config.yaml file"
                    )
    parser.add_argument("--fcgr_read_length",
                        type=int,
                        default=FCGR_params['filters']['read_length'],
                        help="minimal read length to include such as 10000. Default is in config.yaml file"
                        )
    parser.add_argument("--fcgr_consider_tags",
                        type=str,
                        default= FCGR_params['filters']['tags']['consider_tags'],
                        help="Pacbio tags to filter by. Default is 'ec' (effective coverage). Can also include 'np,fn,rq' (number of full-length subreads, forward number of complete passes, Predicted average read quality)"
                        )
    parser.add_argument("--fcgr_tags_filter", "--fcgr_tags_threshold",
                    type=str,
                    default= FCGR_params['filters']['tags']['tags_filter'],
                    help="Thresholds for the tags. Default is '10' (ec=10). Can also be something like '10,5,2,0.999' (ec=10, np=5, fn=2, rq=0.999). Needs to match the tags given in the 'consider_tags' argument"
                    )

### FCGR method:
    parser.add_argument("--IPD_stride",
                        type=str,
                        default=FCGR_params["methods"]["IPD_stride"],
                        choices=["normal", "mean", "max", "emph_A"],
                        help="method for collecting the IPDs (considering the k-mers). Default is in config.yaml file. See file CGR2FCGR.py for more details."
                        )
    parser.add_argument("--IPD_r",
        type=float,
        default=FCGR_params["methods"]["IPD_r"],
        help="step size during the stride. recommended value is either 0.5 or 1. Default is in config.yaml file."
    )
    parser.add_argument("--IPD_stratification",
                        type=str,
                        default=FCGR_params["methods"]["IPD_stratification"],
                        choices=["MeanStd", "FullRange"],
                        help="method for creating the FCGR array. Default is 'MeanStd'."
                        )

### Saving options:
    parser.add_argument("--save_as_npy",
                        type=bool,
                        default=params['save_format']['as_npy'],
                        help="if True, the output will be saved as a .npy file. Default is in config.yaml file"
                        )
    parser.add_argument("--save_as_npz",
                        type=bool,
                        default=params['save_format']['as_npz'],
                        help="if True, the output will be saved as a .npz file (and also a txt file with the names of the long reads). Default is in config.yaml file"
                        )
    parser.add_argument("--plot_examples",
                        type=bool,
                        default=params['save_format']['plot_examples'],
                        help="if True, a few examples will be plotted. Default is in config.yaml file"
                        )

### Parse arguments:
    args = parser.parse_args()
    return args


def get_tags_filter(args, FCGR=False):
    """
    :param args: arguments from the command line
    :return: a dictionary with the tags and their thresholds
    """
    if FCGR:
        tags_to_consider = args.fcgr_consider_tags.split(',')
        tags_thresholds = [float(x) for x in args.fcgr_tags_filter.split(',')]
    else:
        tags_to_consider = args.consider_tags.split(',')
        tags_thresholds = [float(x) for x in args.tags_filter.split(',')]
    assert len(tags_to_consider) == len(tags_thresholds), f"The number of tags and the number of thresholds are different. There are {len(tags_to_consider)} tags and {len(tags_thresholds)} thresholds." \
                                                          f"To change the tags, use the --consider_tags flag. To change the thresholds, use the --tags_filter flag. The flags and values should be comma-separated without spaces."
    tags_filter = dict(zip(tags_to_consider, tags_thresholds))
    return tags_filter


def get_parent_folder(isolate, kmer_size, read_length, tags_filter, output_folder):
    # 1) get the output folder for the isolate/
    full_path = os.path.join(output_folder, isolate)
    # 2) get the CGR folder within the isolate folder, with an ID.
    ID1 = f"l{read_length}_ec{tags_filter['ec']}" if 'ec' in tags_filter.keys() else f"l{read_length}"
    CGR_folder = os.path.join(full_path, f"CGR_{ID1}")
    return CGR_folder


def create_output_folders(CGR_folder, tags_filter, read_length, kmer_size, IPD_stride, IPD_r, channels, IPD_stratification):
    # 1) create the FCGR folder within the CGR folder, with an ID.
    # make ID and a string with the ID explained:
    ID2_explained =  f"kmer: {kmer_size}" \
                     f"IPD stride: {IPD_stride} \n" \
                     f"IPD r: {IPD_r} \n" \
                     f"channels: {channels} \n" \
                     f"IPD stratification: {IPD_stratification} \n" \
                     f"read length: {read_length}"

    ID2 = f"{kmer_size}_{IPD_stride}_{IPD_r}_{channels}_" \
          f"{IPD_stratification}_{read_length}"
    # add the 'ec' tag if it exists:
    # TODO: add the other tags as well
    if 'ec' in tags_filter:
        ID2_explained += f" \n" \
                         f"effective coverage: {round(tags_filter['ec'],1)}"
        ID2 += f"_{round(tags_filter['ec'],1)}"

    # create the folder:
    FCGR_folder = os.path.join(CGR_folder, f"FCGR_{ID2}")
    os.makedirs(FCGR_folder, exist_ok=True)
    # dump the ID_explained to a txt file:
    with open(os.path.join(FCGR_folder, "ID_explained.txt"), 'w') as file:
        file.write(ID2_explained)

    # 2) create the config file within the FCGR folder:
    recreate_config_file(params, FCGR_folder)

    return FCGR_folder


def recreate_config_file(params, folder_path):
    """
    This function recreates the config file from the given parameters.
    """
    file_path = os.path.join(folder_path, f'FCGR_config.yaml')
    if params['mode'] == 'isolate':
        params = {key: value for key, value in params.items() if "community" not in key}
    elif params['mode'] == 'community':
        params = {key: value for key, value in params.items() if "isolate" not in key}
    # the params are cleaner now, so we can dump them to the config file:
    with open(file_path, 'w') as file:
        documents = yaml.dump(params, file)

#%%
if __name__ == '__main__':
    # use config.yaml file to set parameters
    config_path = Parser(params=None, get_config=True) # some path, like: "../config.yaml"
    with open(config_path, 'r') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)

    args = Parser(params)
    # Relevant for FCGR process:
    IPD_stride = args.IPD_stride
    IPD_r = args.IPD_r
    kmer_size = args.kmer_size
    IPD_stratification = args.IPD_stratification
    channels = args.channels
    save_as_npy = args.save_as_npy
    save_as_npz = args.save_as_npz
    plot_examples = args.plot_examples
    read_length = args.fcgr_read_length
    tags_filter = get_tags_filter(args, FCGR=True)

    # Relevant to locate folder:
    output_folder = args.output_folder
    desired_isolate = args.isolate_name
    prev_tags_filter = get_tags_filter(args)
    prev_read_length = args.read_length
    CGR_folder = get_parent_folder(desired_isolate, kmer_size, prev_read_length, prev_tags_filter, output_folder)

    if read_length<prev_read_length:
        print(f"The new read_length ({read_length}) is smaller than the read_length used to create the CGR array ({prev_read_length}). The read_length will be set to {prev_read_length}")
        read_length = prev_read_length
    FCGR_folder = create_output_folders(CGR_folder, tags_filter, read_length, kmer_size, IPD_stride, IPD_r, channels, IPD_stratification)

    # get the CGR array (and the read lengths and ecs), if it exists
    CGR_arrays, read_lengths, ecs, read_names = get_CGR(CGR_folder)
    # filter the CGR array by the tags_filter and the read_length (if there's any difference):
    # TODO: tags_filter is a dictionary with the tags and their thresholds. compare the values of the tags for each key with the previous values.
    if read_length > prev_read_length or tags_filter['ec'] > prev_tags_filter['ec']:
        print(f"Filtering the CGR array by the tags_filter and the read_length")
        CGR_arrays, read_lengths, ecs, read_names = filter_CGR(CGR_arrays, read_lengths, ecs, read_names, tags_filter, read_length)

    # create the FCGR array:
    main(CGR_arrays, read_names, read_lengths, ecs,
         FCGR_folder, IPD_stride, IPD_r, channels, IPD_stratification,
         save_as_npy, save_as_npz, plot_examples)
