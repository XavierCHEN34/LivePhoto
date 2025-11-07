import torch
import os
import argparse
from diffsynth import ModelManager, WanVideoPusaPipeline, save_video, VideoData
import datetime

def main():
    parser = argparse.ArgumentParser(description="Pusa T2V: Text-to-Video Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint file.")
    parser.add_argument("--lora_alpha", type=float, default=1.4, help="Alpha value for LoRA.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
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

    pipe = WanVideoPusaPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # Text-to-video
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        height=720, width=1280, num_frames=81,
        seed=0, tiled=True,
        cfg_scale=args.cfg_scale
    )

    # Create timestamp for the filename
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.lightx2v:
        video_filename = os.path.join(args.output_dir, f"t2v_output_{timestamp}_lightx2v.mp4")
    else:
        video_filename = os.path.join(args.output_dir, f"t2v_output_{timestamp}.mp4")
    print(f"Saved to {video_filename}")
    save_video(video, video_filename, fps=25, quality=5)

if __name__ == "__main__":
    main()
