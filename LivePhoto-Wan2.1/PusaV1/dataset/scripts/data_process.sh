DATASET_PATH="dataset/Vidgen"

CUDA_VISIBLE_DEVICES="0,1,2,3" srun python ./dataset/scripts/preprocess.py \
  --dataset_path "${DATASET_PATH}" \
  --text_encoder_path model_zoo/models_t5_umt5-xxl-enc-bf16.pth \
  --dataloader_num_workers 4 \
  --vae_path model_zoo/Wan2.1_VAE.pth \
  --sam_ckpt_path lang-segment-anything/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --gdino_model_ckpt_path lang-segment-anything/grounding-dino-base \
  --gdino_processor_ckpt_path lang-segment-anything/grounding-dino-base \