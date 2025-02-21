#!/bin/bash
STUDY="brca"
Genes=("PIK3CA" "TP53" "TTN" "CDH1" "GATA3" "MUC16" "KMT2C" "MAP3K1" "SYNE1" "FLG")
for gene in ${Genes[@]};
do
    CUDA_VISIBLE_DEVICES=1 python main.py \
    --lr 1e-4\
    --epochs 32\
    --num_classes 1\
    --model AMIL\
    --features_path /data2/LSL/TCGA_DATA/features/TCGA_${STUDY^^}/RetCCL/pt_files/\
    --csv_file ./M4_csv/1gene/${STUDY}/${gene}.csv\
    --save_path ./results/${STUDY^^}/AMIL/${gene}
done
