#!/bin/bash -l
#SBATCH --account=intro_vsc37047
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00
#SBATCH --mail-user=maarten.laureyssen@student.kuleuven.be
#SBATCH --job-name=KannaBERT-finetuning

module load Python/3.11.5-GCCcore-13.2.0

source /data/leuven/370/vsc37047/miniconda3/etc/profile.d/conda.sh
conda activate wv

export HF_HOME="$VSC_SCRATCH/.cache/huggingface"

echo "training KannaBERT..."
python finetuning/run_classification.py \
--dataset_name "Davlan/sib200" \
--dataset_config_name "eng_Latn" \
--text_column_names "text" \
--label_column_name "category" \
--model_name_or_path "./KannaBERT-xl" \
--cache_dir "$VSC_SCRATCH/.cache/huggingface" \
--output_dir "/scratch/leuven/370/vsc37047/models/KannaBERT-sib200-en" \
--overwrite_output_dir \
--num_train_epochs 10 \
--do_train \
--do_eval \
--do_predict \
--report_to "none" \
--seed 5

echo "training MBERT..."
python finetuning/run_classification.py \
--dataset_name "Davlan/sib200" \
--dataset_config_name "eng_Latn" \
--text_column_names "text" \
--label_column_name "category" \
--model_name_or_path "google-bert/bert-base-multilingual-cased" \
--cache_dir "$VSC_SCRATCH/.cache/huggingface" \
--output_dir "/scratch/leuven/370/vsc37047/models/MBERT-sib200-en" \
--overwrite_output_dir \
--num_train_epochs 10 \
--do_train \
--do_eval \
--do_predict \
--report_to "none" \
--seed 5

echo "testing KannaBERT"
python finetuning/run_classification.py \
--dataset_name "Davlan/sib200" \
--dataset_config_name "kan_Knda" \
--text_column_names "text" \
--label_column_name "category" \
--model_name_or_path "/scratch/leuven/370/vsc37047/models/KannaBERT-sib200-en" \
--cache_dir "$VSC_SCRATCH/.cache/huggingface" \
--output_dir "/scratch/leuven/370/vsc37047/models/KannaBERT-sib200-test" \
--overwrite_output_dir \
--do_predict \
--report_to "none"

echo "testing MBERT"
python finetuning/run_classification.py \
--dataset_name "Davlan/sib200" \
--dataset_config_name "kan_Knda" \
--text_column_names "text" \
--label_column_name "category" \
--model_name_or_path "/scratch/leuven/370/vsc37047/models/MBERT-sib200-en" \
--cache_dir "$VSC_SCRATCH/.cache/huggingface" \
--output_dir "/scratch/leuven/370/vsc37047/models/MBERT-sib200-test" \
--overwrite_output_dir \
--do_predict \
--report_to "none"

echo "finished"
