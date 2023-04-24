#%% Documentation
"""
This script takes a bam file (either a community or an isolate) and creates a folder with CGR files (numpy arrays).

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
The CGR_config.yaml file contains the parameters used to create the CGR files.
The CGR arrays have 4 columns: " [X], [Y], [IPDs], [base] " (the IPDs are processed and normalized but are before the "stride" step).

# Running the code:
1) activate the environment: 'source /home/labs/zeevid/tomerant/FCGR_project/pip_envs/preprocess_env/bin/activate'
2) Activate modules: module load Pysam
3) Edit the config.yaml file to your needs (the values will be taken from there if not stated when calling the function)

# Under the hood:
1) read the config.yaml file and parse the arguments.
2) create the output folder if it doesn't exist (save an edited version of the config file there, based on the arguments that were given).
3) create the CGR files for the isolate or community.
4) Save the CGR files in the output folder.

# imprtant arguments:
    - hyperparams: read_length, ec
    - isolate_or_community: 'isolate' or 'community'
    if 'isolate':
        - isolates_folder: default is '/home/labs/zeevid/Data/Samples/2022-HGR/HybridIsolates/PRJNA231221'
        - desired_isolate: the name of the isolates (in the bam file) that you want to create the CGR files for.
        - output_folder: default is '/home/labs/zeevid/Analyses/2022-HGR/SyntheticCommunity/FCGR' (was '/home/labs/zeevid/Analyses/2022-HGR/HybridIsolates/IPD_CGR')

"""

#%% imports
import os # for creating the output folder and reading files
import numpy as np # for saving the numpy arrays
import pysam # for reading the bam file
import yaml # for reading the config file (is .yaml file)
import argparse # for parsing the arguments when sending from the command line
#from FCGR_parser import config_parsing, Parser  # for parsing the config file (local file)
# from Feature_Engineering.FCGR_parser import config_parsing, Parser  # for parsing the config file (local file)
from tqdm import tqdm # for progress bar
import sys

#%% Parse arguments

def Parser(params, get_config=False):
    parser = argparse.ArgumentParser(description='parsing arguments from config file and command line')
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
        default=params['isolate_or_community'],
        help="Isolate or Community Mode. Default is 'isolate'."
    )
    # else, get these two arguments and continue parsing
    simple_args, unknown = parser.parse_known_args()

    ### Path parsing:
    mode = simple_args.isolate_or_community
    if mode=="isolate":
        parser.add_argument("--home_path", "--isolates_path",
            type=str,
            default=params['isolate_paths']['home_path'],
            help="The folder path in which the isolates are saved. Default is in config.yaml file"
        )
        # isolate name, one of the folder in the isolates_folder
        parser.add_argument("--isolate_name", "--isolate", "--desired_isolate","-i",
            type=str,
            default=params['isolate_paths']['isolate_name'],
            help="Either the name of the isolate, or 'all'. Must be provided if the mode is 'isolate'."
        )
        # output_folder: default is '/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR' (was '/home/labs/zeevid/Analyses/2022-HGR/HybridIsolates/IPD_CGR')
        parser.add_argument("--output_folder", "--output_path",
            type=str,
            default=params['isolate_paths']['output_folder'],
            help="The folder path in which the output folder of the isolate will be saved. Default is in config.yaml file"
        )

    elif mode=="community":
        parser.add_argument("--home_path", "--parent_path",
            type=str,
            default=params['community_paths']['home_path'],
            help="path to home/parent directory"
        )
        parser.add_argument("--N", "-N",
            type=int,
            default=params['community_paths']['N'],
            help="size of the community. Default is in config.yaml file"
        )
        parser.add_argument("--S", "-S",
            type=int,
            default=params['community_paths']['S'],
            help="Seed number of community"
        )
        parser.add_argument("--bam_folder", "--bam_path",
            type=str,
            default=params['community_paths']['bam_folder'],
            help="Folder name within the 'home_path' (such as 'BAM_files'). The bam file is in it. Default is in config.yaml file"
        )
        parser.add_argument("--output_folder", "--output_path",
            type=str,
            default=params['community_paths']['output_folder'],
            help="Folder name within the 'home_path' (such as 'FCGR_files'). The output file will be saved there. Default is in config.yaml file"
        )
    else:
        raise ValueError(f"The mode must be either 'isolate' or 'community'. Got {mode} instead.")

    parser.add_argument("--read_length",
        type=int,
        default=params['filters']['read_length'],
        help="minimal read length to include such as 10000. Default is in config.yaml file"
    )
    parser.add_argument("--max_num_reads",
        type=int,
        default=params['filters']['max_num_reads'],
        help="A maximal cap on the number of reads (out of the bam file) to include in the CGRs. Default is in config.yaml file"
    )
    parser.add_argument("--consider_tags",
                        type=str,
                        default= params['filters']['consider_tags'], #'ec,np,fn,rq', # effective coverage, number of full-length subreads, forward number of complete passes, Predicted average read quality
                        help="Pacbio tags to filter by. Default is 'ec' (effective coverage). Can also include 'np,fn,rq' (number of full-length subreads, forward number of complete passes, Predicted average read quality)"
                        )
    parser.add_argument("--tags_filter", "--tags_threshold",
                        type=str,
                        default=params['filters']['tags_filter'],#'10,5,2,0.999',
                        help="Thresholds for the tags. Default is '10' (ec=10). Can also be something like '10,5,2,0.999' (ec=10, np=5, fn=2, rq=0.999). Needs to match the tags given in the 'consider_tags' argument"
                        )

    ### Saving options parsing:
    parser.add_argument("--save_as_npy",
        type=bool,
        default=params['save_format']['save_as_npy'],
        help="if True, the output will be saved as a .npy file. Default is in config.yaml file"
    )
    parser.add_argument("--save_as_npz",
        type=bool,
        default=params['save_format']['save_as_npz'],
        help="if True, the output will be saved as a .npz file (and also a txt file with the names of the long reads). Default is in config.yaml file"
    )

    ### Parse arguments:
    args = parser.parse_args()
    return args

def check_sys_argv_content(arguments):
    """
    The input is a list generated by sys.argv.
    The function looks for arguments that are for parsing (start with -- or -).
    It returns a list of the arguments that are for parsing (remove the -- or -)
    """
    given_arguments = []
    for arg in arguments:
        if arg.startswith("-"):
            given_arguments.append(arg.lstrip("-"))
    return given_arguments

def update_params(args, params, given_args):
    """
    If an argument was given (value isn't as appears in the config file), update the params dictionary
    Class RecordGiven is used to record the given arguments. The config file wasn't used if args.value.was_given==True
    """
    # update the params dictionary:
    params_sections = ['filters', 'save_format']
    params_sections += ['isolate_paths'] if getattr(args,"isolate_or_community") == 'isolate' else ['community_paths']
    for section in params_sections: # iterate over the sections in the params file
        for arg in given_args: # if the argument was given, update the params file
            if arg in params[section].keys():
                params[section][arg] = getattr(args, arg)

    return params


def get_tags_filter(args):
    """
    :param args: arguments from the command line
    :return: a dictionary with the tags and their thresholds
    """
    tags_to_consider = args.consider_tags.split(',')
    tags_thresholds = [float(x) for x in args.tags_filter.split(',')]
    assert len(tags_to_consider) == len(tags_thresholds), f"The number of tags and the number of thresholds are different. There are {len(tags_to_consider)} tags and {len(tags_thresholds)} thresholds." \
                                                          f"To change the tags, use the --consider_tags flag. To change the thresholds, use the --tags_filter flag. The flags and values should be comma-separated without spaces."
    tags_filter = dict(zip(tags_to_consider, tags_thresholds))
    return tags_filter

#%% Bam processing functions:
def IPD_normalization(IPDs):
    """
    IPDs range between 0 and 255 (int8).
    We transform them into int16 to add 1 to all of them.
    Then we can do log transformation (no zeros) and subtract the mean of the new vector (centering).
    This step is necessary to reduce batch effects.
    """
    IPDs = np.int16(IPDs)
    log_vec = np.log(IPDs+1)
    centered_vec = log_vec - np.mean(log_vec)
    return centered_vec

def bam_iteration(aln, read_length_threshold, tags_filter):
    #aln = next(bam_iterator, -1)
    # if isinstance(aln, int):
    #     return "done"
    name = aln.query_name
    sequence = aln.query
    d = dict(aln.tags)
    # tags: page 14 of www.pacb.com/wp-content/uploads/SMRT_Tools_Reference_Guide_v10.1.pdf
    IPDs = np.array(d['fi'])
    condition = (len(sequence) > read_length_threshold) and (IPDs.size == len(sequence))
    for tag in [*tags_filter]:
        condition = condition and (d[tag] > tags_filter[tag])
    if not condition:
        return "skip"

    # normalize IPDs
    IPDs = IPD_normalization(IPDs = np.array(d['fi']))
    return_list = [sequence, IPDs, name, len(sequence)]
    return_list = return_list+[d['ec']] if 'ec' in [*tags_filter] else return_list
    return return_list

#%% CGR walk
def next_step(x_coord, y_coord, next_base):
    step_direction = {"A":[1,1], "C":[-1,1], "G":[-1,-1], "T":[1,-1]}
    x_coord = (x_coord + step_direction[next_base][0])/2
    y_coord = (y_coord + step_direction[next_base][1])/2
    return x_coord, y_coord

def CGRwalk(sequence,IPDs):
    """
    sequence: string
    IPDs: numpy array
    """
    X_CGR = []
    Y_CGR = []
    # initialize coordinates
    x_coord, y_coord, z_coord = (0,0,0)
    # walk
    for i, base in enumerate(sequence):
        x_coord, y_coord = next_step(x_coord, y_coord, base)
        # append coordinates
        X_CGR.append(x_coord)
        Y_CGR.append(y_coord)

    # convert to numpy array of 4 columns
    bases_as_int = list(sequence.encode('utf-8'))
    # CGR_array = np.array([X_CGR, Y_CGR, IPDs, bases_as_int], dtype=np.float32).T
    CGR_array = np.array([X_CGR, Y_CGR, IPDs, bases_as_int]).T

    return CGR_array

#%% Paths and file/folder creation
def isolate_paths(desired_isolate, isolates_folder):
    # check that the desired isolate is in the folder (if not requested all isolates). If not, raise error.
    assert desired_isolate in os.listdir(isolates_folder), f"{desired_isolate} not in {isolates_folder}"
    ccs_folder = os.path.join(isolates_folder, desired_isolate, 'ccs_hifikinetics_allkinetics')
    bam_file = [file for file in os.listdir(ccs_folder) if file.endswith("sequelII_ccs_hifikinetics_allkinetics.bam")]
    # check that there is a bam file in the folder. If not, raise error.
    assert len(bam_file) > 0, f"no bam file in {ccs_folder}"
    bam_file = bam_file[0]
    bam_path = os.path.join(ccs_folder, bam_file)
    return bam_path

def recreate_config_file(params, folder_path):
    """
    This function recreates the config file from the given parameters.
    But if a different value was given (the default value given in the config wasn't used),
    it will be changed by update_params
    """

    # create the config file
    file_path = os.path.join(folder_path, f'CGR_config.yaml')
    # since this is only for the CGR, remove any keys with "FCGR" in them.
    params = {key: value for key, value in params.items() if "FCGR" not in key}
    # depending on the mode, remove the relevant keys.
    if params['isolate_or_community'] == 'isolate':
        params = {key: value for key, value in params.items() if "community" not in key}
    elif params['isolate_or_community'] == 'community':
        params = {key: value for key, value in params.items() if "isolate" not in key}
    # now we have clean params, we can dump them into the config file.
    with open(file_path, 'w') as file:
        documents = yaml.dump(params, file)


def create_output_folders(isolate, params,  tags_filter, read_length, output_folder = "/home/labs/zeevid/Analyses/2022-HGR/SyntheticCommunity/FCGR"):
    """
    :param isolate: isolate name (string)
    :param params: loaded yaml file (dict)
    :param  tags_filter: used for the effective coverage, called using tags_filter['ec']
    :return: creates folders and config file
    """
    # 1) create the output folder for the isolate/
    full_path = os.path.join(output_folder, isolate)
    os.makedirs(full_path, exist_ok=True)
    print(f"output folder for {isolate} is {full_path}")
    # 2) create a CGR folder within the isolate folder, with an ID.
    ID = f"l{read_length}_ec{tags_filter['ec']}" if 'ec' in tags_filter.keys() else f"l{read_length}"
    CGR_folder = os.path.join(full_path, f"CGR_{ID}")
    os.makedirs(CGR_folder, exist_ok=True)
    print(f"Created folder {CGR_folder}")
    # 3) create a config file within the CGR folder
    recreate_config_file(params, CGR_folder)

    return CGR_folder

#%% Save CGR
def save_CGR_to_npy(CGR_array, read_name, output_folder):
    """
    This is done per read, and saves the CGR array as npy file.
    :param CGR_array: CGR array of the read (4 columns)
    :param read_name: name of the read
    :param output_folder: output folder
    :return: saves the CGR array as npy file
    """
    np.save(os.path.join(output_folder, f"{read_name}.npy"), CGR_array)

def save_CGR_to_npz(CGR_array_dict, output_folder, file_name="CGR_array"):
    """
    This is done for all the reads, and saves the CGR array as npz file.
    The npy arrays within the npz are: CGR array, actual read length, actual ec.
    Also, a txt file is saved with the read names.
    :param CGR_array_dict: dictionary of CGR arrays. Keys are: CGR_arrays, read_lengths, ecs, read_names
    :param file_name: name of the file
    :param output_folder: output folder
    :return: saves the CGR array as npy file
    """
    # 1) save the npy arrays into an npz file
    # CGR_arrays = np.array([CGR_array_dict[read_name][0] for read_name in CGR_array_dict.keys()])
    # read_lengths = np.array([CGR_array_dict[read_name][1] for read_name in CGR_array_dict.keys()])
    # ecs = np.array([CGR_array_dict[read_name][2] for read_name in CGR_array_dict.keys()])

    # drop the read_names from the dictionary
    read_names = CGR_array_dict.pop('read_names')
    CGR_array_dict['CGR_arrays'] = np.array(CGR_array_dict['CGR_arrays'], dtype=object) # convert to object array because of different length of arrays
    CGR_array_dict['read_lengths'] = np.array(CGR_array_dict['read_lengths'], dtype=np.uint16)
    CGR_array_dict['ecs'] = np.array(CGR_array_dict['ecs'], dtype=np.float16)
    np.savez(os.path.join(output_folder, f"{file_name}.npz"), **CGR_array_dict)

    #np.savez(os.path.join(output_folder, f"{file_name}.npz"), CGR_arrays = CGR_arrays, read_lengths = read_lengths, ecs = ecs, dtype=object)
    # print message
    print(f"Saved {file_name}.npz to {output_folder}")
    # 2) save the read names
    with open(os.path.join(output_folder, f"{file_name}_read_names.txt"), "w") as fp:
        fp.write('\n'.join(read_names))
    # print message
    print(f"Saved {file_name}_read_names.txt to {output_folder}")

#%% Main Function
def main(bam_iterator, read_length, tags_filter,max_num_reads, output_folder, save_as_npy=False, save_as_npz=True, do_tqdm = False):
    counter = 0
    array_dict = {}
    pbar = tqdm(bam_iterator) if do_tqdm else bam_iterator # progress bar
    for aln in pbar: # aln is a pysam.AlignedSegment object
        iteration = bam_iteration(aln, read_length, tags_filter) # get the sequence, IPDs and name of the read
        # if iteration is type str, it means that the read was not processed
        if isinstance(iteration, str) and iteration == "skip": # if any of the conditions are not met
            continue
        # otherwise, it's a list. check the length of it.
        # if it's 4, then: [sequence, IPDs, name, actual_read_length].
        if len(iteration) == 4:
            sequence, IPDs, name, actual_read_length = iteration
            name = name.replace("/", "_") # the name might include forward slashes, turn them into underscores
            read_name = f"{name}__LEN{actual_read_length}"
            ec = np.nan
        # if it's 5: it's [sequence, IPDs, name, actual_read_length, ec]
        else: #elif len(iteration) == 5:
            sequence, IPDs, name, actual_read_length, ec = iteration
            name = name.replace("/", "_") # the name might include forward slashes, turn them into underscores
            read_name = f"{name}__LEN{actual_read_length}_EC{round(ec,1)}"


        CGR_array = CGRwalk(sequence, IPDs) # get the CGR array
        ### SAVING ###
        if save_as_npy: # save the CGR array as npy file in the output folder
            save_CGR_to_npy(CGR_array, read_name, output_folder) # save the CGR array as npy file
            if counter==0:
                print(f"Saved {read_name}.npy to {output_folder}")
        if save_as_npz: # construct a dictionary of CGR arrays
            # if the dictionary is empty, create the first key
            if array_dict == {}:
                array_dict['CGR_arrays'] = [CGR_array]
                array_dict['read_lengths'] = [actual_read_length]
                array_dict['ecs'] = [ec]
                array_dict['read_names'] = [read_name]
            else: # if the dictionary is not empty, add the new key
                array_dict['CGR_arrays'].append(CGR_array)
                array_dict['read_lengths'].append(actual_read_length)
                array_dict['ecs'].append(ec)
                array_dict['read_names'].append(read_name)

        counter += 1
        if counter == max_num_reads:
            break
        if do_tqdm:
            pbar.set_postfix({"Processed reads": counter})

    if save_as_npz:
        save_CGR_to_npz(array_dict, output_folder)
    return


#%% Main
if __name__ == '__main__':
    # use config.yaml file to set parameters
    # TODO: use correct method of assigning path to config.yaml file
    config_path = Parser(params=None, get_config=True)
    with open(config_path, 'r') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)

    args = Parser(params)
    given_arguments = check_sys_argv_content(sys.argv)
    params = update_params(args, params, given_arguments) # update the params file with the new parameters
    # convert args to variables
    tags_filter = get_tags_filter(args)
    read_length = args.read_length
    max_num_reads = args.max_num_reads
    isolate_or_community = args.isolate_or_community # 'isolate' or 'community'
    save_as_npy = args.save_as_npy
    save_as_npz = args.save_as_npz
    # TODO: implement the community option
    #if isolate_or_community == 'isolate':
    isolates_folder =  args.home_path # default is '/home/labs/zeevid/Data/Samples/2022-HGR/HybridIsolates/PRJNA231221'
    desired_isolate = args.isolate_name # Relevant for 'isolate' mode. the name of the isolate. In isolate mode can be some folder name that appears in isolates_folder or 'all'
    output_folder = args.output_folder # default is '/home/labs/zeevid/Analyses/2022-HGR/SyntheticCommunity/FCGR' (was '/home/labs/zeevid/Analyses/2022-HGR/HybridIsolates/IPD_CGR')

    # the desired_isolate can be 'all' or a specific isolate name (folder name within isolates_folder)
    if desired_isolate != 'all':
        # create output folders
        CGR_folder = create_output_folders(desired_isolate, params, tags_filter, read_length, output_folder)
        # get the bam file path
        bam_path = isolate_paths(desired_isolate, isolates_folder)
        # create the bam iterator
        bam_iterator = pysam.AlignmentFile(bam_path, check_sq=False)
        # run the main function
        main(bam_iterator, read_length, tags_filter,max_num_reads, CGR_folder, save_as_npy, save_as_npz)

    else:
        for isolate in os.listdir(isolates_folder):
            # create output folders
            CGR_folder = create_output_folders(isolate, params, tags_filter, read_length, output_folder)
            # get the bam file path
            bam_path = isolate_paths(isolate, isolates_folder)
            # create the bam iterator
            bam_iterator = pysam.AlignmentFile(bam_path, check_sq=False)
            # run the main function
            main(bam_iterator, read_length, tags_filter,max_num_reads, CGR_folder, save_as_npy, save_as_npz)


    #channels = args.channels
    #flatten = args.flatten
    #home_path = args.home_path
    # bam_path = os.path.join(home_path, args.bam_folder, f"{args.file_name}.bam")
    # if not os.path.exists(bam_path):
    #     print(f"bam file {bam_path} does not exist")
    #     exit(1)
    # file_name = f"{args.file_name}_transform_{channels}Ch.npz" if args.transform else f"{args.file_name}_{channels}Ch.npz"
    # output_path = os.path.join(home_path, args.output_folder, file_name)
    #
    #
    #transform = args.transform
    # get the tags and their thresholds from the strings inputted by the user
    # tags_filter needs to be a dictionary where the keys are the tags and the values are the thresholds
    # if the number of tags is different from the number of thresholds, raise an error with a message


    # if not os.path.exists(output_path) or args.redo_if_exists:
    #
    #
    #     bam_iterator = pysam.AlignmentFile(bam_path, check_sq=False)
    #     array_dict = main(bam_iterator, kmer_size, read_length, tags_filter, channels, transform)
    #
    #     #np.savez(output_path, **array_dict)
    #
    #     niceSave(array_dict, output_path)
    # # file name structrue is community_S-{S}_N-{N}
    # # extract the seed and the number of isolates from the file name:
    # S = int(args.file_name.split('_')[1].split('-')[1])
    # N = int(args.file_name.split('_')[2].split('-')[1])
    # #call_Split4DeepDPM(N, S, channels)


# def getFullOutPath(bam_path, output_path, file_type="npz"):
#     community = bam_path.split("/")[-1].split(".")[0]
#     out_path = os.path.join(output_path, f"{community}.{file_type}")
#     return out_path

# def save_to_dict(FCGR_array, array_dict, name):
#     array_dict[name] = FCGR_array
#     return array_dict
#
# def turnLabelsToNumbers(labels):
#     unique_labels = np.unique(labels)
#     number_labels = np.zeros(labels.shape).astype(int)
#     for i in range(len(unique_labels)):
#         number_labels[labels == unique_labels[i]] = i
#     return number_labels
#
# def niceSave(array_dict, output_path):
#     """
#     two arrays are saved: one is an array of all the FCGR arrays, (n, 2**kmer_size, 2**kmer_size, 3)
#     another is an array of all the names of the reads.
#     :param array_dict:
#     :param output_path:
#     :return:
#     """
#     FCGR_array = np.array(list(array_dict.values()))
#     names_array = np.array(list(array_dict.keys()))
#     labels = np.array([name.split("__LABEL:__")[-1] for name in names_array])
#     number_labels = turnLabelsToNumbers(labels)
#     # if the number of unique number labels is greater than 100, arise an error
#     if len(np.unique(number_labels)) > 100:
#         raise ValueError(f"the number of unique labels is {len(np.unique(number_labels))}, which is greater than 100. Expected less.")
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     np.savez(output_path, FCGR_array=FCGR_array, unique_ids=names_array, labels=labels, number_labels=number_labels)
#     print(f"saved to {output_path}")
