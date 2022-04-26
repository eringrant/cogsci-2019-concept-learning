#!/bin/bash

set -e

if (( $# != 3 ))
then
  echo "Usage: scripts/run_human_comp.sh SUMMARY_DIR SAVE_DIR DATASET_SOURCE_DIR"
  exit 1
fi

arch_opts="large_conv_binary_maml"
num_updates_opts="1"
meta_step_size_opts="0.001"
meta_batch_size_opts="64"
meta_clip_norm_opts="10."
update_batch_size_opts="5"
update_step_size_opts="0.01"
meta_optimizer_opts="adam_9"
update_optimizer_opts="sgd"
train_dataset_opts="hier_unary_ilsvrc"
test_dataset_opts="dummy"  # because of `human_comp`

sha=$(git log --pretty=format:'%h' -n 1)
date=`date +%Y-%m-%d`

python scripts/main.py \
--human_comp \
--num_train_iters 50000 \
--num_train_batches 480000 \
--num_val_batches 10000 \
--num_test_batches 100000 \
--num_classes 1 \
--summary_interval 1 \
--save_interval 1000 \
--input_size 84 \
--init xavier \
--model_arch $arch_opts \
--num_updates $num_updates_opts \
--meta_step_size $meta_step_size_opts \
--meta_batch_size $meta_batch_size_opts \
--meta_clip_norm $meta_clip_norm_opts \
--update_step_size $update_step_size_opts \
--update_training_batch_size $update_batch_size_opts \
--update_validation_batch_size $update_batch_size_opts \
--meta_optimizer $meta_optimizer_opts \
--update_optimizer $update_optimizer_opts \
--train_dataset $train_dataset_opts \
--val_dataset $train_dataset_opts \
--test_dataset $test_dataset_opts \
--prefix $test_dataset_opts \
--summary_dir $1 \
--save_dir $2 \
--data_source_dir $3
