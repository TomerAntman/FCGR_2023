"""
python multiple_cgrs_send.py FCGR \
MycobacteriumKansasii_SAMN23242884 NocardiaAsteroides_SAMN23242890 NocardiaYamanashiensis_SAMN23242892 EnterobacterHormaechei_SAMN16357575 CitrobacterBraakii_SAMN16357563 MycobacteriumOstraviense_SAMN23242882 ClostridiumInnocuum_SAMN22091681 AcinetobacterNosocomialis_SAMN16357537 --channels 1

Things to play with:
IPD stride: ["normal", "mean", "max", "emph_A"]
IPD r: [0.5, 1]
channels: [1, 3]
IPD stratification: ["MeanStd", "FullRange"]
things I'm less likely to change:
kmer: 5
read length: 10000
effective coverage: 10.0
"""
import subprocess
import sys
from itertools import product
import os

def sendjob_fCGR(isolate_name, FCGR=True, argument_tupples=None):
    outputpath=f"/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Isolates/{isolate_name}/bsub_output.txt"
    errorpath=f"/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Isolates/{isolate_name}/bsub_error.txt"
    command = f"bsub -R rusage[mem=32G] -n 1 -q new-short -o {outputpath} -e {errorpath} " \
              f"\" module load Pysam ; source /home/labs/zeevid/tomerant/FCGR_project/pip_envs/preprocess_env/bin/activate ; " \
              f"cd /home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/FCGR_2023/Feature_Engineering ; "
    if FCGR:
        command += f"python CGR2FCGR.py --isolate_name {isolate_name} "
    else:
        command += f"python Bam2CGR.py --isolate_name {isolate_name} "

    if argument_tupples is not None:
        for argument, value in argument_tupples:
            command += f" {argument} {value}"
    command += "\""
    print(command)
    subprocess.call(command, shell=True)



def sendjob_community(N, S,is_balanced, IPD_stride, IPD_r, channels, IPD_stratification, read_length=10000, ec=10.0, kmer=5):
    FCGR_folder_name = f"FCGR_{kmer}_{IPD_stride}_{IPD_r}_{channels}_{IPD_stratification}_{read_length}_{ec}"
    balance = "balanced" if is_balanced else "unbalanced"

    outputpath=f"/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Communities/N{N}_S{S}/bsub_OUTPUT_{FCGR_folder_name}_{balance}.txt"
    errorpath=f"/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Communities/N{N}_S{S}/bsub_ERROR_{FCGR_folder_name}_{balance}.txt"

    command = f"bsub -R rusage[mem=32G] -n 1 -q new-short -o {outputpath} -e {errorpath} " \
              f"\" source /home/labs/zeevid/tomerant/FCGR_project/pip_envs/preprocess_env/bin/activate ; " \
              f"cd /home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/FCGR_2023/Feature_Engineering ; "

    command += f"python FCGR_community_creation.py --N {N} --S {S} --IPD_stride {IPD_stride} --IPD_r {IPD_r} --channels {channels} --IPD_stratification {IPD_stratification} "
    #command += f" --read_length {read_length} --ec {ec} --kmer {kmer} "
    if not is_balanced:
        command += "--unbalanced "
    command += "\""
    subprocess.call(command, shell=True)


def hyperparameter_options():
    IPD_stride = ["normal", "mean", "max", "emph_A"]
    IPD_r = [0.5, 1.0]
    channels = [1, 3]
    IPD_stratification = ["MeanStd", "FullRange"]
    return product(IPD_stride, IPD_r, channels, IPD_stratification)

def make_argument_tupples(hps):
    argument_tupples = []
    for IPD_stride, IPD_r, channels, IPD_stratification in hps:
        argument_tupples.append(("--IPD_stride", IPD_stride))
        argument_tupples.append(("--IPD_r", IPD_r))
        argument_tupples.append(("--channels", channels))
        argument_tupples.append(("--IPD_stratification", IPD_stratification))
    return argument_tupples

if __name__ == '__main__':
    hps = hyperparameter_options()
    # if sys.argv[1] is int then it indicates a community:
    if sys.argv[1].isdigit():
        N = int(sys.argv[1])
        S = int(sys.argv[2])
        is_balanced = sys.argv[3]
        for IPD_stride, IPD_r, channels, IPD_stratification in hps:
            ###
            community_path = f"/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Communities/N{N}_S{S}/"
            folder_name=f"FCGR_5_{IPD_stride}_{IPD_r}_{channels}_{IPD_stratification}_10000_10.0"
            if folder_name in [i for i in os.listdir(community_path) if not i.endswith(".txt")]:
                continue
            ###
            print(f"###\n!!! Sending job for N={N}, S={S}, is_balanced={is_balanced}, IPD_stride={IPD_stride}, IPD_r={IPD_r}, channels={channels}, IPD_stratification={IPD_stratification} !!!\n###")
            sendjob_community(N, S, is_balanced, IPD_stride, IPD_r, channels, IPD_stratification)
    else:
        isolate_names = sys.argv[1:]
        for IPD_stride, IPD_r, channels, IPD_stratification in hps:
            argument_tupples = make_argument_tupples([(IPD_stride, IPD_r, channels, IPD_stratification)])
            for isolate_name in isolate_names:
                #print(argument_tupples)
                sendjob_fCGR(isolate_name, FCGR=True, argument_tupples=argument_tupples)



#%%
