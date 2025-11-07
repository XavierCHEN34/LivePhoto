from PIL import Image
import torch
import os
from diffsynth import ModelManagerWan22, Wan22VideoPusaPipeline, save_video # TODO
# from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
import datetime  # Add this import for datetime functionality
import argparse

def main():
    parser = argparse.ArgumentParser(description="Pusa T2V: Text-to-Video Generation with Wan2.2 model")
    
    # --- Arguments ---
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the output video.")
    
    # LoRA arguments
    parser.add_argument("--high_lora_path", type=str, required=True, help="Path to the high noise LoRA checkpoint file.")
    parser.add_argument("--high_lora_alpha", type=float, default=1.3, help="Alpha value for high noise LoRA.")
    parser.add_argument("--low_lora_path", type=str, required=True, help="Path to the low noise LoRA checkpoint file.")
    parser.add_argument("--low_lora_alpha", type=float, default=1.4, help="Alpha value for low noise LoRA.")
    
    # Model paths
    parser.add_argument("--clip_model_path", type=str, default="/scratch/dyvm6xra/dyvm6xrauser02/AIGC/Wan2.1-I2V-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", help="Path to CLIP model.")
    parser.add_argument("--high_model_dir", type=str, default="/scratch/dyvm6xra/dyvm6xrauser02/AIGC/Wan2.2-T2V-A14B/high_noise_model", help="Directory of high noise model.")
    parser.add_argument("--low_model_dir", type=str, default="/scratch/dyvm6xra/dyvm6xrauser02/AIGC/Wan2.2-T2V-A14B/low_noise_model", help="Directory of low noise model.")
    parser.add_argument("--base_dir", type=str, default="/scratch/dyvm6xra/dyvm6xrauser02/AIGC/Wan2.2-T2V-A14B", help="Base directory for other models like T5 and VAE.")
    
    # Pipeline parameters
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--switch_DiT_boundary", type=float, default=0.875, help="Boundary to switch DiT.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    model_manager = ModelManagerWan22(device="cpu")

    # Load CLIP model
    model_manager.load_models(
        [args.clip_model_path],
        torch_dtype=torch.float32,  # Image Encoder is loaded with float32
    )

    # Get all safetensors files from the checkpoint directory
    high_model_files = sorted([os.path.join(args.high_model_dir, f) for f in os.listdir(args.high_model_dir) if f.endswith('.safetensors')])
    low_model_files = sorted([os.path.join(args.low_model_dir, f) for f in os.listdir(args.low_model_dir) if f.endswith('.safetensors')])

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

    model_manager.load_loras_wan22(args.high_lora_path, lora_alpha=args.high_lora_alpha, model_type="high")

    model_manager.load_loras_wan22(args.low_lora_path, lora_alpha=args.low_lora_alpha, model_type="low")

    pipe = Wan22VideoPusaPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    # Text-to-video
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        height=args.height, width=args.width, num_frames=args.num_frames,
        seed=args.seed, tiled=True, switch_DiT_boundary=args.switch_DiT_boundary,
        cfg_scale=args.cfg_scale
    )

    # Create timestamp for the filename
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # A more robust way to get parts of the path
    try:
        high_lora_parts = os.path.normpath(args.high_lora_path).split(os.sep)
        low_lora_parts = os.path.normpath(args.low_lora_path).split(os.sep)
        check_point_name = f"{high_lora_parts[-5]}_{high_lora_parts[-3]}_{low_lora_parts[-5]}_{low_lora_parts[-3]}"
    except IndexError:
        # Fallback to a simpler name if path structure is not as expected
        check_point_name = "custom_lora"

    if args.lightx2v:
        video_filename = os.path.join(args.output_dir, f"wan22_T2V_video_{check_point_name}_{timestamp}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}_lightx2v.mp4")
    else:
        video_filename = os.path.join(args.output_dir, f"wan22_T2V_video_{check_point_name}_{timestamp}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}.mp4")
    print(f"saved to {video_filename}")
    save_video(video, video_filename, fps=24, quality=5)


if __name__ == "__main__":
    main()

