<div align="center">

# FlowMotion: Training-Free Flow Guidance for Video Motion Transfer

[Zhen Wang](https://zhenwang97.github.io/)<sup>1</sup>, Youcan Xu<sup>2</sup>, Jun Xiao<sup>2</sup>, [Long Chen](https://zjuchenlong.github.io/)<sup>1</sup>*  

<sup>1</sup>The Hong Kong University of Science and Technology (HKUST)  
<sup>2</sup>Zhejiang University  

**CVPR 2026**


<a href="https://arxiv.org/abs/2603.06289">
<img src='https://img.shields.io/badge/arxiv-FlowMotion-blue' alt='Paper PDF'></a>



### Efficient Video Motion Transfer with one single RTX 3090 🚀  ###

<div align="center">
  <img src="assets/teaser.png"  width="100%">
</div>

</div>

## 🚀 Quick Start


### 1. Setup

Create the environment and install the dependencies by running:

```
conda create -n flowmotion python=3.10
conda activate flowmotion
pip install -r requirements.txt
```


### 2. Data Preparation

Put your videos in the `data/` folder, we have already prepared several examples.


### 3. Running FlowMotion

Run the following command for video motion transfer:

```
python run.py --video_path ./data/49f/bus.mp4 --target_prompt "A black shark is moving through crystal-clear ocean waters."
```


### Try more examples, such as: ###

```
python run.py --video_path ./data/49f/car-turn.mp4 --target_prompt "A dog is running in a city garden."

python run.py --video_path ./data/49f/blackswan.mp4 --target_prompt "A spaceship glides silently through the universe."

python run.py --video_path ./data/49f/hike.mp4 --target_prompt "A penguin is walking along a frozen coastline."
```

### Try longer videos with more frames, such as: ###

```
python run.py --video_path ./data/81f/car-turn.mp4 --target_prompt "A tiger is running through a dense jungle."

python run.py --video_path ./data/81f/bus.mp4 --target_prompt "A eagle soaring over rugged mountain peaks."
```


### 4. Variant of Source Motion Representation

For videos with more complex motions, try using the varient of source motion representation with `clean latent`, such as:

```
python run.py --video_path ./data/49f/motocross-jump.mp4 --target_prompt "A monkey on a motorcycle is mid-jumping on a dirt track in the forest." --guidance_type clean_latent
```

besides, `larger learning rate` and `more detailed prompt` all help to get better results, such as:

```
python run.py --video_path ./data/49f/snowboard.mp4 --target_prompt "A monkey is surfing down a towering, sun-bleached dune, a sleek figure of focused agility against the vast emptiness. A plume of golden sand arcs from the board, shimmering in the heat haze that dances above the sand. In the distance, a cluster of resilient cacti stands witness to this burst of dynamic life under the relentless sun." --guidance_type clean_latent --lr_base 0.01
```



## 💻 VRAM Requirements

Below is the VRAM usage on a single NVIDIA RTX 3090 (24GB VRAM), you can save more cost by using `enable_model_cpu_offload()`.


| Video Length | VRAM Usage | Enable CPU Offload |
| :----------: | :--------------: | :----------------: |
| 49 frames    | 19 GB            | 4.9 GB             |
| 81 frames    | 20 GB            | 5.8 GB             |








## 🗓️ TODO
- [x] Release demo
- [ ] Release benchmark  




## 🖊️ BibTeX
If you find this project useful in your research, please consider cite:

```bibtex
@article{wang2026flowmotion,
  title={FlowMotion: Training-Free Flow Guidance for Video Motion Transfer},
  author={Wang, Zhen and Xu, Youcan and Xiao, Jun and Chen, Long},
  journal={arXiv preprint arXiv:2603.06289},
  year={2026}
}
```

## Acknowledgements
We thank to [Wan](https://github.com/Wan-Video/Wan2.1) and [diffusers](https://github.com/huggingface/diffusers).