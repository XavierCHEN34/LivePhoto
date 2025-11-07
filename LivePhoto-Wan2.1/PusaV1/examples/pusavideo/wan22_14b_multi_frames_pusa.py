from PIL import Image
import torch
import os
import sys
import argparse
from diffsynth import ModelManagerWan22, Wan22VideoPusaMultiFramesPipeline, save_video
import datetime

def main():
    parser = argparse.ArgumentParser(description="Pusa Conditional Video Generation from one or more images using dual DiT models.")
    parser.add_argument("--image_paths", type=str, nargs='+', required=True, help="Paths to one or more conditioning image frames.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt for video generation.")
    parser.add_argument("--cond_position", type=str, required=True, help="Comma-separated list of frame indices for conditioning. You can use any position from 0 to 20.")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames. A value of 0 means the condition image is used as totally clean, higher value means add more noise.")    
    parser.add_argument("--base_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B", help="Directory of the T2V model components (T5, VAE).")
    parser.add_argument("--high_noise_model_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/high_noise_model", help="Directory of the high noise DiT model components.")
    parser.add_argument("--low_noise_model_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/low_noise_model", help="Directory of the low noise DiT model components.")

    parser.add_argument("--high_lora_path", type=str, required=True, help="Path to the high noise LoRA checkpoint file.")
    parser.add_argument("--high_lora_alpha", type=float, default=1.4, help="Alpha value for high noise LoRA.")
    parser.add_argument("--low_lora_path", type=str, required=True, help="Path to the low noise LoRA checkpoint file.")
    parser.add_argument("--low_lora_alpha", type=float, default=1.4, help="Alpha value for low noise LoRA.")
    
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--switch_DiT_boundary", type=float, default=0.875, help="Boundary to switch from high noise DiT to low noise DiT.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("Loading models...")
    model_manager = ModelManagerWan22(device="cpu")

    # Load DiT, VAE, and Text Encoder models
    high_model_files = sorted([os.path.join(args.high_noise_model_dir, f) for f in os.listdir(args.high_noise_model_dir) if f.endswith('.safetensors')])
    low_model_files = sorted([os.path.join(args.low_noise_model_dir, f) for f in os.listdir(args.low_noise_model_dir) if f.endswith('.safetensors')])
    
    model_manager.load_models(
        [
            high_model_files,
            low_model_files,
            os.path.join(args.base_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(args.base_dir, "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16,
    )
    if args.lightx2v:
        # Lighx2v for acceleration
        high_lora_path = "./model_zoo/PusaV1/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
        model_manager.load_loras_wan22_lightx2v(high_lora_path, model_type="high")
        low_lora_path = "./model_zoo/PusaV1/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
        model_manager.load_loras_wan22_lightx2v(low_lora_path, model_type="low")

    # Load LoRAs
    model_manager.load_loras_wan22(args.high_lora_path, lora_alpha=args.high_lora_alpha, model_type="high")
    model_manager.load_loras_wan22(args.low_lora_path, lora_alpha=args.low_lora_alpha, model_type="low")
    
    pipe = Wan22VideoPusaMultiFramesPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)
    print(f"Models loaded successfully")

    cond_pos_list = [int(x.strip()) for x in args.cond_position.split(',')]
    noise_mult_list = [float(x.strip()) for x in args.noise_multipliers.split(',')]
    
    images = []
    target_w, target_h = 1280, 720
    for p in args.image_paths:
        img = Image.open(p).convert("RGB")
        original_w, original_h = img.size

        ratio = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)

        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        background = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        background.paste(img_resized, (paste_x, paste_y))
        images.append(background)

    if len(images) != len(cond_pos_list) or len(images) != len(noise_mult_list):
        raise ValueError("The number of --image_paths, --cond_position, and --noise_multipliers must be the same.")

    multi_frame_images = {
        cond_pos: (img, noise_mult) 
        for cond_pos, img, noise_mult in zip(cond_pos_list, images, noise_mult_list)
    }

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        multi_frame_images=multi_frame_images,
        num_inference_steps=args.num_inference_steps,
        height=720, width=1280, num_frames=81,
        seed=0, tiled=True,
        switch_DiT_boundary=args.switch_DiT_boundary,
        cfg_scale=args.cfg_scale,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.lightx2v:
        video_filename= os.path.join(args.output_dir, f"wan22_multi_frame_output_{timestamp}_cond_{str(cond_pos_list)}_noise_{str(noise_mult_list)}_high_alpha_{args.high_lora_alpha}_low_alpha_{args.low_lora_alpha}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}_lightx2v.mp4")
    else:
        video_filename = os.path.join(args.output_dir, f"wan22_multi_frame_output_{timestamp}_cond_{str(cond_pos_list)}_noise_{str(noise_mult_list)}_high_alpha_{args.high_lora_alpha}_low_alpha_{args.low_lora_alpha}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}.mp4")

    print(f"Saved to {video_filename}")
    save_video(video, video_filename, fps=24, quality=5)

if __name__ == "__main__":
    main() 