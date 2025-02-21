#!/bin/bash

# STUDIES=("brca" "crc" "luad" "ucec" "gbmlgg")
STUDIES=("brca")
Num_expert=(5)

for STUDY in ${STUDIES[@]};
do
  for num_expert in ${Num_expert[@]};
  do
    CUDA_VISIBLE_DEVICES=2 python main.py \
    --multi \
    --model M4 \
    --num_expert $num_expert \
    --lr 1e-4\
    --epochs 32\
    --num_classes 10\
    --features_path /data2/LSL/TCGA_DATA/features/TCGA_${STUDY^^}/RetCCL/pt_files/\
    --csv_file ./M4_csv/10genes/all_${STUDY}.csv\
    --save_path ./results/${STUDY^^}/M4
  done 
done 
