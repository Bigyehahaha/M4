#!/bin/bash

STUDIES=("luad")
Num_expert=(5)

for STUDY in ${STUDIES[@]};
do
  for num_expert in ${Num_expert[@]};
  do
    CUDA_VISIBLE_DEVICES=4 python main.py \
    --multi \
    --model MMoE\
    --num_expert $num_expert\
    --lr 1e-4\
    --epochs 32\
    --num_classes 10\
    --features_path /data2/LSL/TCGA_DATA/features/TCGA_${STUDY^^}/RetCCL/pt_files/\
    --csv_file ./M4_csv/10genes/all_${STUDY}.csv\
    --save_path ./results/${STUDY^^}/MMoE
  done
done 