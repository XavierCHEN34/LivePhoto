<p align="center">

  <h2 align="center">LivePhoto: Real Image Animation with Text-guided Motion Control</h2>
  <p align="center"> 
        <a href="https://arxiv.org/abs/2312.02928"><img src='https://img.shields.io/badge/arXiv-LivePhoto-red' alt='Paper PDF'></a>
        <a href='https://xavierchen34.github.io/LivePhoto-Page/'><img src='https://img.shields.io/badge/Project_Page-LivePhoto-green' alt='Project Page'></a>
    <br>
    <b>The University of Hong Kong &nbsp; | &nbsp;  Alibaba Group  | &nbsp;  Ant Group </b>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/demo.gif">
    </td>
    </tr>
  </table>

## News
📢 LivePhoto-Wan2.1 is released.

## LivePhoto-Wan2.1

**LivePhoto-Wan2.1** supports text-guided image-to-video generation with control over motion intensity levels. Built upon the **Wan2.1-T2V-1.3B** architecture, it is adapted for I2V tasks using **Pusa** fine-tuning strategy. A **motion intensity module** is plugged in to adjust movement strength in the generated videos.

  <table align="center">
    <tr>
    <td>
      <img src="assets/teaser2.gif" width="60%">
    </td>
    </tr>
    <tr>
    <td>
      <img src="assets/teaser1.gif" width="60%">
    </td>
    </tr>
  </table>

### Installation
```
conda create -n livephoto python=3.10 -y
conda activate livephoto
cd ./LivePhoto-Wan2.1/PusaV1
pip install -e .
pip install xfuser>=0.4.3 absl-py peft lightning pandas deepspeed wandb av 
```

### Model Preparation
```
pip install -U "huggingface_hub[cli]==0.34.0"
hf download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth --local-dir ./model_zoo/Wan2.1/base/
hf download Wan-AI/Wan2.1-T2V-1.3B --include="google/*" --local-dir ./model_zoo/Wan2.1/base
hf download shirley430316/LivePhoto-Wan2.1 lora.safetensors base.safetensors --local-dir ./model_zoo/Wan2.1/
```

After proper preparation, the directory looks like:
```
./model_zoo
  - Wan2.1
    - base
      - Wan2.1_VAE.pth
      - models_t5_umt5-xxl-enc-bf16.pth
      - google
    - base.safetensors
    - lora.safetensors
```
### Usage Example
#### I2V with Motion Intensity Levels
```
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
```

### Limitations
- Video generation quality is inherently limited by the capabilities of the base Wan2.1-T2V-1.3B model, e.g., camera motion control is not currently supported.
- Certain image types are likely to generate low quality videos, e.g. cartoon and animated contents, possibly due to dataset biases.

### Acknowledgement
This version is developed upon the codebase of [Pusa-VidGen](https://github.com/Yaofang-Liu/Pusa-VidGen). Much appreciation for the insightful project.

## Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{chen2023livephoto,
    title={LivePhoto: Real Image Animation with Text-guided Motion Control},
    author={Chen, Xi and Liu, Zhiheng and Chen, Mengting and Feng, Yutong and Liu, Yu and Shen, Yujun and Zhao, Hengshuang},
    journal={arXiv preprint arXiv:2312.02928},
    year={2023}
    }
```
