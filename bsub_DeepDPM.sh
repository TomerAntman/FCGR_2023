#!/bin/bash
IPD_stride="max"
IPD_r=1.0
IPD_stratif="MeanStd"
read_length=10000
ec="10.0"
channels=3
AE_archit="LinAE"
latent_dim=30
transformation="normalize" # divide_by_max, normalize
N=8
S=99
kmer=5
is_balanced='balanced'
number_for_logger=0

exp_name=$channels"_"$AE_archit"__LD"$latent_dim"_Trans-"$transformation
output_dir="/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Communities/N"$N"_S"$S"/FCGR_"$kmer"_"$IPD_stride"_"$IPD_r"_"$channels"_"$IPD_stratif"_"$read_length"_"$ec"/"$is_balanced"/Clustering_Results/results_LD"$latent_dim"_"$AE_archit"/"

outputpath=$output_dir"/"$exp_name"_outputs.txt"
errorpath=$output_dir"/"$exp_name"_errors.txt"

bsub -q gpu-short -J multiclass -gpu gmodel=NVIDIAGeForceRTX2080Ti:num=1:j_exclusive=no -R rusage[mem=40000] -R affinity[thread*6] -o $outputpath -e $errorpath "source /home/labs/testing/class49/DeepDPM/DDPM_env/bin/activate ; sh /home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/FCGR_2023/DeepDPM_run.sh"
