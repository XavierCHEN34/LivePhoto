from PIL import Image
import torch
import os
import sys
import argparse
from diffsynth import ModelManager, PusaMultiFramesPipeline, save_video
import datetime

def main():
    parser = argparse.ArgumentParser(description="Pusa Conditional Video Generation from one or more images (Image-to-Video, Start-End-Frame-to-Video, Multi-Frames-to-Video).")
    parser.add_argument("--image_paths", type=str, nargs='+', required=True, help="Paths to one or more conditioning image frames.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--cond_position", type=str, required=True, help="Comma-separated list of frame indices for conditioning. You can use any position from 0 to 20")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames. A value of 0 means the condition image is used as totally clean, higher value means add more noise. For I2V, you can use 0.2 or any from 0 to 1. For Start-End-Frame, you can use 0.2,0.7, or any from 0 to 1.")
    parser.add_argument("--i2v_model_path", type=str, default="model_zoo/PusaV1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", help="Path to the I2V CLIP model.")
    parser.add_argument("--t2v_model_dir", type=str, default="model_zoo/Wan2.1/base", help="Directory of the T2V model components.")
    parser.add_argument("--dit_path", type=str, default="model_zoo/Wan2.1/base.safetensors", help="Path of the DiT model with motion intensity module.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint file.")
    parser.add_argument("--lora_alpha", type=float, default=1.4, help="Alpha value for LoRA.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    parser.add_argument("--motion_intensity", type=int, default=3, help="Motion intensity levels from 1 to 6.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("Loading models...")
    model_manager = ModelManager(device="cpu")
    
    base_dir = args.t2v_model_dir
    model_files = args.dit_path
    
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
    
    pipe = PusaMultiFramesPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)
    print(f"Models loaded successfully")

    cond_pos_list = [int(x.strip()) for x in args.cond_position.split(',')]
    noise_mult_list = [float(x.strip()) for x in args.noise_multipliers.split(',')]
    
    images = [Image.open(p).convert("RGB").resize((1280, 720), Image.LANCZOS) for p in args.image_paths]

    if len(images) != len(cond_pos_list) or len(images) != len(noise_mult_list):
        raise ValueError("The number of --image_paths, --cond_position, and --noise_multipliers must be the same.")

    multi_frame_images = {
        cond_pos: (img, noise_mult) 
        for cond_pos, img, noise_mult in zip(cond_pos_list, images, noise_mult_list)
    }

    video = pipe(
        prompt=args.prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        multi_frame_images=multi_frame_images,
        num_inference_steps=args.num_inference_steps,
        height=480, width=832, num_frames=81,
        seed=0, tiled=True,
        cfg_scale=args.cfg_scale,
        motion_intensity=args.motion_intensity
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(args.output_dir, f"output_{timestamp}_alpha_{args.lora_alpha}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}_intensity_{args.motion_intensity}.mp4")
    print(f"Saved to {video_filename}")
    save_video(video, video_filename, fps=25, quality=5)

if __name__ == "__main__":
    main()