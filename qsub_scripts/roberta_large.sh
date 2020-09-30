#$ -S /bin/bash

#$ -N ROBERT_LARGE
#$ -j y

#$ -l tmem=16G
#$ -l h_rt=72:00:00
#$ -l gpu=true,gpu_v100=yes
# $ -pe gpu 2
#$ -R y

#$ -cwd

export PATH=/share/apps/python-3.7.2-shared/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:${LD_LIBRARY_PATH}

source /share/apps/examples/source_files/cuda/cuda-10.1.source

export RACE_DIR=../../RACE
python ../transformers-examples/run_multiple_choice.py \
--task_name race \
--model_name_or_path roberta-large \
--do_train \
--do_eval \
--do_predict \
--data_dir $RACE_DIR \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--max_seq_length 512 \
--output_dir ../output/roberta_large-race \
--per_gpu_eval_batch_size=2 \
--per_gpu_train_batch_size=2 \
--gradient_accumulation_steps 8 \
--overwrite_output \
