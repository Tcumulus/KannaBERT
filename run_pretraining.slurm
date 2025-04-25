#!/bin/bash -l
#SBATCH --account=intro_vsc37047
#SBATCH --clusters=genius
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maarten.laureyssen@student.kuleuven.be
#SBATCH --job-name=KannaBERT

module load Python/3.11.5-GCCcore-13.2.0

source /data/leuven/370/vsc37047/miniconda3/etc/profile.d/conda.sh
conda activate wv

export HF_HOME="$VSC_SCRATCH/.cache/huggingface"

python pretraining/run_mlm.py \
--config_name "./pretraining/config.json" \
--tokenizer_name "./tokenizer/out" \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--train_file "./data/knxl.txt" \
--validation_split_percentage 10 \
--line_by_line \
--do_train \
--do_eval \
--num_train_epochs 1 \
--overwrite_output_dir \
--output_dir "./KannaBERT-xl" \
--cache_dir "$VSC_SCRATCH/.cache/huggingface" \
--eval_strategy "epoch" \
--logging_steps 1000 \
--save_steps 10000 \
--save_total_limit 2 \
--fp16 \
--warmup_ratio 0.05 \
--max_seq_length 256
