#!/bin/bash -l
#SBATCH --account=intro_vsc37047
#SBATCH --clusters=genius
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mail-user=maarten.laureyssen@student.kuleuven.be
#SBATCH --job-name=KannaBERT-finetuning

module load Python/3.11.5-GCCcore-13.2.0

source /data/leuven/370/vsc37047/miniconda3/etc/profile.d/conda.sh
conda activate wv

export HF_HOME="$VSC_SCRATCH/.cache/huggingface"

echo "training MBERT..."
python finetuning/run_classification.py \
--train_file "./data/finetuning/sentiment140-en/train_xs.csv" \
--validation_file "./data/finetuning/sentiment140-en/validation_xs.csv" \
--test_file "./data/finetuning/sentiment140-en/test_xs.csv" \
--text_column_names "text" \
--label_column_name "target" \
--model_name_or_path "google-bert/bert-base-multilingual-cased" \
--cache_dir "$VSC_SCRATCH/.cache/huggingface" \
--output_dir "./models/MBERT-sentiment-en-xs" \
--overwrite_output_dir \
--num_train_epochs 10 \
--do_train \
--do_eval \
--do_predict \
--save_total_limit 1 \
--report_to "none"

echo "finished"
