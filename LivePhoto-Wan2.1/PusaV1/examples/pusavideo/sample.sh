CUDA_VISIBLE_DEVICES=0 python examples/pusavideo/wan_14b_multi_frames_pusa.py \
  --image_paths "./demos/input_image.jpg" \
  --prompt "A cute orange kitten with big round eyes stands upright on its hind legs on a smooth wooden floor. The kitten begins to move its tiny front paws up and down rhythmically, swaying its body left and right as if dancing. Its fluffy tail flicks slightly behind it, and the soft lighting creates a warm, cozy indoor atmosphere. The kitten’s ears twitch gently as it keeps its balance, adding to the charm of its playful little dance. The background stays softly blurred, keeping focus on the kitten’s adorable movements." \
  --cond_position "0" \
  --noise_multipliers "0" \
  --lora_path "./model_zoo/Wan2.1/lora.safetensors" \
  --lora_alpha 1.2 \
  --num_inference_steps 30 \
  --cfg_scale 5 \
  --motion_intensity 6  # valid motion intensity levels from 1 through 6