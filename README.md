# UnmaskMe
-[ ] insert sample image here

## What is UnmaskMe

## Installation
To install requirements simply run this within the project root: 
```
pip install -r requirements.txt 
```

## Usage

### Training
#### Generating The Dataset
Download FFHQ thumbnails 128x128 data from this
[link](https://archive.org/download/ffhq-dataset/thumbnails128x128.zip).

#### Pretraing model
Our pretrained model can be found [here]()

#### Training script
```
python train.py --dataroot ./datasets/maskface/data_combined --model pix2pix --name face2mask --print_freq 50 --num_threads 12
```
### Inference
```
python test.py --dataroot ./../our_data/data_combined --model pix2pix --name face2mask --direction BtoA --results_dir ./results/
```

## Useful resources
- CycleGAN & pix2pix [repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- MaskTheFace [repo](https://github.com/aqeelanwar/MaskTheFace)

