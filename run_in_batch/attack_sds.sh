# export EXPERIMENT_NAME=$data_id
export EXPERIMENT_NAME=$EXPERIMENT_NAME
export MODEL_PATH=$model_path
export CLEAN_TRAIN_DIR="$data_path/$dataset_name/$data_id/set_A" 
export CLEAN_ADV_DIR="$data_path/$dataset_name/$data_id/set_B"
# export OUTPUT_DIR="outputs/simac/$dataset_name/$EXPERIMENT_NAME"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$wandb_run_name"
export CLASS_DIR=$class_dir


# ------------------------- Train ASPL on set B -------------------------
mkdir -p $OUTPUT_DIR
rm -r $OUTPUT_DIR/* 2>/dev/null || true
cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean_ref
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

# base_set="""
# attack:
#     epsilon: 16
#     steps: 100
#     input_size: 512
#     mode: sds
#     img_path: /data/home/yekai/github/DiffAdvPerturbationBench/datasets/VGGFace2-clean/0/set_B
#     output_path: out_vk/
#     alpha: 1
#     g_mode: "-"
#     use_wandb: False
#     project_name: iclr_mist
#     diff_pgd: [False, 0.2, 'ddim100'] # un-used feature
#     using_target: False
#     target_rate: 5
#     device: 0
#     max_exp_num: 100
# """
# mist deviation correction (SDS code base on mist)
r=$((r - 1))

mist_cmd="""python code/diff_mist_DAPB.py \
--model_path=$MODEL_PATH  \
--input_dir_path=$CLEAN_ADV_DIR \
--output_path=$OUTPUT_DIR \
--epsilon=$r \
--steps=$attack_steps \
--concept_prompt='$concept_prompt' \
--model_config=$model_config \
--mode=$mode \
--g_mode=$g_mode \
--target_rate=$target_rate \
--input_size=$input_size \
--alpha=$alpha \
--diff_pgd=$diff_pgd \
--using_target=$using_target \
--max_exp_num=$max_exp_num \

"""
echo $mist_cmd
eval $mist_cmd




