#! /usr/bin/bash

set -e

repo_dir=/iblm/netapp/data4/dlaub/projects/hyena-dna
data_dir=/iblm/netapp/data4/shared_dir/hyena_dna_collab/downstream_tasks/iPSCORE_cvpc
gvl=$data_dir/gvl/cvpc_f25k_l2p17_top5k.gvl
bed=$data_dir/bed_files/genes_Flank25k_Seq17_top5k.chunk.bed
vcf=/iblm/netapp/data4/Frazer_collab/ipscs/datasets/raw/genotypes/michigan_impute/results/final_vcf/merged_sorted.vcf.gz
bw=$data_dir/sample2bw.csv
length=$((2**17))
max_jitter=$((2**10))

genvarloader $gvl $bed --vcf "$vcf" --bigwig-table "$bw" --length $length --max-jitter $max_jitter --overwrite

python $repo_dir/scripts/add_transformed_intervals.py --squashed-tmm --log-tmm --standard-scaler $data_dir/gvl/scalers