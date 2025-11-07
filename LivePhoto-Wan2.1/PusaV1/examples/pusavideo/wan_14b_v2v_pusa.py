from PIL import Image
import torch
import os
import sys
import argparse
from diffsynth import ModelManager, PusaV2VPipeline, save_video, VideoData
import datetime
import cv2

def process_video_frames(video_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate scaling and cropping parameters
    target_width = 1280
    target_height = 720
    target_ratio = target_width / target_height
    original_ratio = width / height
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize maintaining aspect ratio
        if original_ratio > target_ratio:
            # Video is wider than target
            new_width = int(height * target_ratio)
            # Crop width from center
            start_x = (width - new_width) // 2
            frame = frame[:, start_x:start_x + new_width]
        else:
            # Video is taller than target
            new_height = int(width / target_ratio)
            # Crop height from center
            start_y = (height - new_height) // 2
            frame = frame[start_y:start_y + new_height]
        
        # Resize to target dimensions
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Pusa V2V: Video-to-Video Generation")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the conditioning video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--cond_position", type=str, required=True, help="Comma-separated list of frame indices for conditioning.")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint file.")
    parser.add_argument("--lora_alpha", type=float, default=1.4, help="Alpha value for LoRA.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("Loading models...")
    model_manager = ModelManager(device="cpu")

    base_dir = "model_zoo/PusaV1/Wan2.1-T2V-14B"
    model_files = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.safetensors')])
    
    model_manager.load_models(
        [
            model_files,
            os.path.join(base_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(base_dir, "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16,
    )
    
    if args.lightx2v:
        # Lightx2v for acceleration
        lightx2v_lora_path = "./model_zoo/PusaV1/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
        model_manager.load_lora_lightx2v(lightx2v_lora_path)
        # lightx2v_lora_path = "./model_zoo/PusaV1/Wan2.1-LightX2V/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors"
        # lightx2v_lora_alpha = 1.0
        # model_manager.load_lora(lightx2v_lora_path,lora_alpha=lightx2v_lora_alpha)
    
    model_manager.load_lora(args.lora_path, lora_alpha=args.lora_alpha)
    
    pipe = PusaV2VPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)
    print(f"Models loaded successfully")

    cond_pos_list = [int(x.strip()) for x in args.cond_position.split(',')]
    noise_mult_list = [float(x.strip()) for x in args.noise_multipliers.split(',')]

    conditioning_video = process_video_frames(args.video_path)

    video = pipe(
        prompt=args.prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        conditioning_video=conditioning_video,
        conditioning_indices=cond_pos_list,
        conditioning_noise_multipliers=noise_mult_list,
        num_inference_steps=args.num_inference_steps,
        height=720, width=1280, num_frames=81,
        seed=0, tiled=True,
        cfg_scale=args.cfg_scale
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.basename(args.video_path).split('.')[0]
    if args.lightx2v:
        video_filename = os.path.join(args.output_dir, f"v2v_{output_filename}_{timestamp}_cond_{str(cond_pos_list)}_noise_{str(noise_mult_list)}_alpha_{args.lora_alpha}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}_lightx2v.mp4")
    else:
        video_filename = os.path.join(args.output_dir, f"v2v_{output_filename}_{timestamp}_cond_{str(cond_pos_list)}_noise_{str(noise_mult_list)}_alpha_{args.lora_alpha}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}.mp4")
    print(f"Saved to {video_filename}")
    save_video(video, video_filename, fps=25, quality=5)

if __name__ == "__main__":
    main()