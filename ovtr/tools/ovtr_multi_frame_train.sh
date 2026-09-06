CUDA_DEVICES="0, 1, 2, 3"
MASTER_PORT=9982
NPROC_GPU=4
PRETRAIN_MODEL="../model_zoo/ovtr_det_pretrain.pth"
OUTPUT="./weights"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=${NPROC_GPU} \
    --use_env \
    ./main.py \
    --config_file ./config/ovtr_5_frame_train_val.py \
    --dataset_file lvis_generated_img_seqs \
    --epochs 1 \
    --with_box_refine \
    --two_stage \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_drop 13 \
    --pretrain ${PRETRAIN_MODEL} \
    --num_workers 4 \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 1 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 2 3 4 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --max_len 250 \
    --output_dir ${OUTPUT}
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=${NPROC_GPU} \
    --use_env \
    ./main.py \
    --config_file ./config/ovtr_5_frame_train_val.py \
    --dataset_file lvis_generated_img_seqs \
    --epochs 16 \
    --with_box_refine \
    --two_stage \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --lr_drop 13 \
    --resume ${OUTPUT}/checkpoint0000.pth \
    --num_workers 4 \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 1 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 2 3 4 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --max_len 250 \
    --output_dir ${OUTPUT}