### Data Preprocess

1. Prepare your dataset

Download your dataset with the following directory structure:
```bash
dataset/
├─ scripts/
└─ your_dataset/
    ├─ train/
    │  ├─ train_video.mp4
    │  └─ ...
    └─ metadata.csv # [file_name, text]
```

2. Install [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything) for segmentation.

3. Config and run `./dataset/scripts/data_process.sh`