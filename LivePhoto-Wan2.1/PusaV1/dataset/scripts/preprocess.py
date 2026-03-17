from turtle import back
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPusaPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import wandb
import datetime
from lang_sam import LangSAM

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        if self.is_i2v:
            video, first_frame = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, langsam_path=None, cotracker_path=None, neg_lab_path=None, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPusaPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.langsam = LangSAM(sam_type="sam2.1_hiera_tiny", # ["sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_large", "sam2.1_hiera_base_plue"]
                sam_ckpt_path=args.sam_ckpt_path,
                gdino_model_ckpt_path=args.gdino_model_ckpt_path,
                gdino_processor_ckpt_path=args.gdino_processor_ckpt_path,
                device=self.device)
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        
        with open("/userhome/cs3/u3612721/LivePhoto-archive/LivePhoto/LivePhoto-Wan2.1/PusaV1/dataset/scripts/negative_labels.txt", "r") as f:
            self.negative_labels = f.read().strip().split("\n")


    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
                
            else:
                image_emb = {}
                        
            # convert the first frame to PIL image
            first_frame = video[:, :, 0, :, :]
            first_frame = first_frame.squeeze().permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            first_frame = (first_frame * 0.5 + 0.5).clamp(0, 1)  # de-normalize to [0,1]
            first_frame = (first_frame.float().cpu().numpy() * 255).astype(np.uint8)
            first_frame = Image.fromarray(first_frame)
            
            subject_mask = self.get_subject_mask(first_frame, text, self.negative_labels, path).to(self.pipe.device)
            motion_intensity = self.get_motion_intensity(video, subject_mask)
            
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb, "motion_intensity": motion_intensity}
            torch.save(data, path + ".tensors.pth")
    
    def get_subject_mask(self, image, text_prompt, neg_labels, path=None):
        '''
        Get the segmentation mask for the first frame.
        '''
        
        results = self.langsam.predict([image], [text_prompt])

        result = results[0]
        masks = result['masks']
        labels = result['labels']
        scores = result['scores']
        
        combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        
        if type(masks)==list:
            # nothing is detected
            return torch.zeros((1, 1, 480, 832), dtype=torch.uint8)
        
        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            # print(f"Mask {i}: Label={label}, Score={score}")
            
            label_words = label.lower().strip().split()
            
            skip = False
            for neg_label in neg_labels:
                if neg_label in label_words:
                    # print(f"Skipping mask {i} due to negative label match: {neg_label}")
                    skip = True
                    break
            if skip:
                continue

            # Threshold the mask (values > 0.5 become 1, others 0)
            binary_mask = (mask > 0.5).astype(np.uint8)
    
            # Add to combined mask (using logical OR)
            combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)

        # Convert to white (255) where mask exists, black (0) elsewhere
        final_mask = (combined_mask * 255).astype(np.uint8)
        
        # Save the mask image for debugging
        # if path is not None:
        #     mask_image = Image.fromarray(final_mask)  # 'L' mode for grayscale
        #     path = path + "_mask.jpg"
        #     print(f"Saving mask to {path}")
        #     mask_image.save(path)

        mask_tensor = torch.from_numpy(final_mask.copy()).float()    
        # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

        return mask_tensor

    def get_motion_intensity(self, video, subject_mask):
        '''
        Get the motion intensity map for the video.
        '''
        B, C, T, H, W = video.shape
        
        # apply cotracker to the video
        video = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
        video = video.to(self.device)
        subject_mask = subject_mask.to(self.device)
        
        if torch.all(subject_mask == 0):
            # print("No subject detected in the first frame. Skipping motion intensity calculation.")
            return 0
        
        tracks, _ = self.cotracker(
            video,
            grid_size=10,
            grid_query_frame=0,
            backward_tracking=True
        )   # [B, T, num_points, 2]
        
        # fix B=1
        tracks = tracks[0, :]  # (T, num_points, 2)
        
        # find which points are on the subject mask at the first frame
        first_frame_tracks = tracks[0]  # (num_points, 2)
        #print(first_frame_tracks.shape)
        
        first_frame_tracks = first_frame_tracks.long()
        first_frame_mask = subject_mask[0, 0]  # (H, W)

        x_coords = torch.clamp(first_frame_tracks[:, 0], 0, W - 1)  # (num_points,)
        y_coords = torch.clamp(first_frame_tracks[:, 1], 0, H - 1)  # (num_points,)
        track_on_subject = first_frame_mask[y_coords, x_coords] > 0  # (num_points, )
        # print(track_on_subject.shape)
        
        motion_intensity = 0
        for t in range(1, T):
            subject_mean_displacement = torch.mean(tracks[t][track_on_subject] - tracks[t-1][track_on_subject], dim=0)  # (2,)
            background_mean_displacement = torch.mean(tracks[t][~track_on_subject] - tracks[t-1][~track_on_subject], dim=0)  # (2,)
            
            if torch.norm(background_mean_displacement) > torch.norm(subject_mean_displacement):
                # if the background is moving more than the subject, we consider it as camera motion and ignore it.
                continue
            
            motion_intensity += torch.norm(subject_mean_displacement - background_mean_displacement)
        return motion_intensity / (T - 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Whether to use Weights & Biases for logging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--sam_ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint of SAM model for LangSAM.",
    )
    parser.add_argument(
        "--gdino_model_ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint of Grounding DINO model for LangSAM.",
    )
    parser.add_argument(
        "--gdino_processor_ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint of Grounding DINO processor for LangSAM.",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        args.dataset_path + "/metadata.csv",
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)

if __name__ == '__main__':
    args = parse_args()
    data_process(args)