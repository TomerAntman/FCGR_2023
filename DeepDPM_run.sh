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
python -m pdb "/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/FCGR_2023/DeepDPM/DeepDPM_alternations.py" \
  --exp_name $channels"_"$AE_archit"__LD"$latent_dim"_Trans-"$transformation  \
  --dataset "N"$N"_S"$S \
  --dir "/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Communities/N"$N"_S"$S"/FCGR_"$kmer"_"$IPD_stride"_"$IPD_r"_"$channels"_"$IPD_stratif"_"$read_length"_"$ec"/"$is_balanced"/Data/" \
  --output_dir "/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Communities/N"$N"_S"$S"/FCGR_"$kmer"_"$IPD_stride"_"$IPD_r"_"$channels"_"$IPD_stratif"_"$read_length"_"$ec"/"$is_balanced"/Clustering_Results/results_LD"$latent_dim"_"$AE_archit"/" \
  --tag $AE_archit",LatDim_"$latent_dim",easy,balanced,pretrain" \
  --API_key "c184359965e7380901e00bfbfc7c40608302deef" \
  --True_k $N  \
  --gpus 0  \
  --latent_dim $latent_dim  \
  --pretrain_epochs 350  \
  --lr 0.002  \
  --epoch 200  \
  --max_epochs 400  \
  --lambda_ 0.05  \
  --beta 0.01  \
  --init_k 10  \
  --NIW_prior_nu 12  \
  --prior_sigma_choice data_std  \
  --prior_sigma_scale 0.0001  \
  --number_of_ae_alternations 3  \
  --transform_input_data $transformation \
  --batch_size 250  \
  --split_merge_every_n_epochs 30  \
  --number_for_logger $number_for_logger  \
  --init_cluster_net_using_centers  \
  --reinit_net_at_alternation  \
  --alternate  \
  --save_model  \
  --save_full_model  \
  --use_labels_for_eval  \
  --is_image  \
  --is_balanced='balanced' \
  --plot_embeddings \
  --pretrain
#_path "/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/FCGR_2023/saved_models/N8_S99/1_LinAE__LD10_Trans-normalize/trained_model_350Epochs" \
#  --pretrain \
  
