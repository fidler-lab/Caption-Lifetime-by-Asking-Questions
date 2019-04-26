# CVPR-code-release

# Contents
1. [Environment Setup](#environment-setup)
2. [Training](#Training)

# Environment Setup
All the code has been run and tested on:

- Python 2.7.15 (coco-caption requires 2.7)
- Pytorch 1.0.0
- CUDA 9.0
- TITAN X/Xp and GTX 1080Ti GPUs

First clone the repository:

```
git clone https://github.com/shenkev/Caption-Images-through-a-Lifetime-by-Asking-Questions.git
```

- Go into the downloaded code directory
- Add the project to PYTHONPATH

```
cd <path_to_downloaded_directory>
export PYTHONPATH=$PWD
```
# Inference

- skip the setup and instead:
  - See section "Download pretrained modules" and follow instructions
  - Download pretrained resnet model from [here](https://drive.google.com/open?id=1vrU-5DsJEHG75EWPMF43VvbwHcwoYbU-) and place in Utils/preprocess/checkpoint


Run demo.py, this will run inference on the file cat.png. Feel feel to use your own image and update the path to file in demo.py.

```
python demo.py
```

# Training setup

## 1. Python dependencies and Stanford NLP
```
chmod +x setup.sh
./setup.sh
```

This will:

- Install python dependencies
- Download Stanford NLP package for parsing part-of-speech
- Download [coco-caption](https://github.com/tylin/coco-caption.git)
- Download [pyciderevalcap](https://github.com/ruotianluo/cider)

## 2. Download images and preprocess them

- Download the images from this [link](http://mscoco.org/dataset/#download). We need the 2014 training images and 2014 val images.
- You should put the train2014/ and val2014/ in a directory of your choice, denoted as `$IMAGE_ROOT`.

- Download pretrained resnet model from [here](https://drive.google.com/open?id=1vrU-5DsJEHG75EWPMF43VvbwHcwoYbU-) and place in Utils/preprocess/checkpoint
- Preprocess images the images by running
```
python Utils/preprocess/preprocess_imgs.py --input_json Data/annotation/dataset_coco.json --output_dir $IMAGE_ROOT/features --images_root $IMAGE_ROOT
```

<b>Warning</b>: the prepro script will fail with the default MSCOCO data because one of their images is corrupted. See this [issue](https://github.com/karpathy/neuraltalk2/issues/4) for the fix, it involves manually replacing one image in the dataset.

## 3. Download training data and preprocessing

- Download training data [here](https://drive.google.com/open?id=1WYSl6SohjhAt0j9SGv26bO9M_amNqU3T)
- Unzip it into Data/annotation
- Precompute indexes for CIDEr

```
python Utils/preprocess/preprocess_cider.py --data_file Data/annotation/cap_train.p --output_file Data/annotation/coco-words
```

- Prepare lifelong learning data splits

```
python Utils/preprocess/preprocess_llsplits.py --data_file Data/annotation/cap_train.p --output_file Data/annotation/train3_split --warmup 3 --num_splits 4 --num_caps 2
```

- You can play with the chunk sizes and # chunks using `warmup` and `num_splits` parameters

# Training

- You can either download trained caption, question generator, VQA modules or train them yourself

## 1. Download pretrained modules
- You can download trained Caption, Question generator, VQA modules
- Download model checkpoints [here](https://drive.google.com/open?id=1xVX4_sw5elDPexQXS5Zy4Him5SQkSkzu)
- Place in Data/model_checkpoints
- The captioning module was trained using 10% warmup data

## 1. Training modules

- Train caption module
- In `Experiments/caption.json` change `exp_dir` to the working directory, `img_dir` to `$IMAGE_ROOT`

```
python Scripts/train_caption3.py --experiment Experiments/caption3.json
```

- Train VQA module
- In `Experiments/vqa.json` change `exp_dir` to the working directory, `img_dir` to `$IMAGE_ROOT`

```
python Scripts/train_vqa.py --experiment Experiments/vqa.json
```

- Train question generator module
- In `Experiments/question3.json` change `exp_dir` to the working directory, `img_dir` to `$IMAGE_ROOT`, `vqa_path` to vqa model checkpoint and `cap_path` to caption model checkpoint
```
python Scripts/train_quegen.py --experiment Experiments/question3.json
```

## 2. Lifelong training

- In `Experiments/lifelong3.json` change `exp_dir` to the working directory, `img_dir` to `$IMAGE_ROOT`, `vqa_path` to vqa model checkpoint and `cap_path` to caption model checkpoint, `quegen_path` to question generator model checkpoint

- You can play with parameters `H, lamda, k`


```
python Scripts/train_lifelong.py --experiment Experiments/lifelong3.json
```

- Track training

```
cd Results/lifelong
tensorboard --logdir tensorboard/
```

-  Visualize qualitative results

```
cd Results/lifelong/lifelong3
```
