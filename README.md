<p align="center">
 <img src="./assets/lumina2-logo.png" width="40%"/>
 <br>
</p>

# 	Lumina-Image 2.0 : An Efficient, Unified and Transparent Image Generative Model
<div align="center">

[![Static Badge](https://img.shields.io/badge/Lumina--Image--2.0%20checkpoints-Model(2B)-yellow?logoColor=violet&label=%F0%9F%A4%97%20Lumina-Image-2.0%20checkpoints)](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)

[![Static Badge](https://img.shields.io/badge/Huiying-6B88E3?logo=youtubegaming&label=Demo%20Lumina-Image-2.0)](https://magic-animation.intern-ai.org.cn/image/create)&#160;
[![Static Badge](https://img.shields.io/badge/Gradio-6B88E3?logo=youtubegaming&label=Demo%20Lumina-Image-2.0)](http://47.100.29.251:10010/)&#160;
</div>







## 📰 News

- [2024-1-25] 🚀🚀🚀 We are excited to release `Lumina-Image 2.0`, including:
  - 🎯 Checkpoints, Fine-Tuning and Inference code.
  - 🎯 Website & Demo are live now! Check out the [Huiying](https://magic-animation.intern-ai.org.cn/image/create) and [Gradio Demo](http://47.100.29.251:10010/)!
- [2024-1-31] 🚀🚀🚀 We have released the latest .pth format weight file [Google Drive](https://drive.google.com/drive/folders/1LQLh9CJwN3GOkS3unrqI0K_q9nbmqwBh?usp=drive_link).

## 🎥 Demo



<div align="center">
  <video src="https://github.com/user-attachments/assets/b1d6dddf-4185-492d-b804-47d3d949adb5" width="70%"> </video>
</div>

## 🎨 Qualitative Performance

![Qualitative Results](assets/Demo.png)





## 📊 Quantatitive Performance
![Quantitative Results](assets/quantitative.png)


## 🎮 Model Zoo


| Resolution | Parameter| Text Encoder | VAE | Download URL  |
| ---------- | ----------------------- | ------------ | -----------|-------------- |
| 1024       | 2B             |    [Gemma-2B](https://huggingface.co/google/gemma-2b)  |   [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae) | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) |

## 💻 Finetuning Code
### 1. Create a conda environment and install PyTorch
```bash
conda create -n Lumina2 -y
conda activate Lumina2
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
### 2.Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Install flash-attn
```bash
pip install flash-attn --no-build-isolation
```
### 4. Prepare data
You can place the links to your data files in `./configs/data.yaml`. Your image-text pair training data format should adhere to the following:
```json
{
    "image_path": "path/to/your/image",
    "prompt": "a description of the image"
}
```
### 5. Start finetuning
```bash
bash scripts/run_1024_finetune.sh
```
## 🚀 Inference Code
We support multiple solvers including Midpoint Solver, Euler Solver, and **DPM Solver** for inference.
> [!Note]
> Both the Gradio demo and the direct inference method use the .pth format weight file, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LQLh9CJwN3GOkS3unrqI0K_q9nbmqwBh?usp=drive_link).
- Gradio Demo
```python   
python demo.py \
    --ckpt /path/to/your/ckpt \
    --res 1024 \
    --port 12123
``` 


- Direct Batch Inference
```bash
bash scripts/sample.sh
```


