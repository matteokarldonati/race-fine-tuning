#$ -S /bin/bash

#$ -N BERT_LARGE
#$ -j y

#$ -l tmem=16G
#$ -l h_rt=100:00:00
#$ -l gpu=true,gpu_v100=yes
# $ -pe gpu 2
#$ -R y

#$ -cwd

export PATH=/share/apps/python-3.7.2-shared/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:${LD_LIBRARY_PATH}

source /share/apps/examples/source_files/cuda/cuda-10.1.source

python3 ../bert-race/run_race.py \
--data_dir=RACE \
--bert_model=bert-large-uncased \
--output_dir=../output/bert_large-race \
--max_seq_length=320 \
--do_train \
--do_eval \
--do_lower_case \
--train_batch_size=8 \
--eval_batch_size=1 \
--learning_rate=1e-5 \
--num_train_epochs=2 \
--gradient_accumulation_steps=8
