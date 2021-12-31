# detex
Explaining object detectors
## Installation
### GPU
```bash
conda env create -n detex -f conda_gpu.yml
conda activate detex
```
### CPU
```bash
conda env create -n detex_cpu -f conda_cpu.yml
conda activate detex_cpu
```

## Download data and annotations
```bash
chmod +x scripts/download_cocoval.sh
./scripts/download_cocoval.sh
```

## Results
### Kernel SHAP
- Using 2000 samples per box in 100 images (254 boxes in total)
- Using 256 superpixels per image as features


### Examples:
Correct detection
![Correct detection](./docs/images/kshap/kshap_2000s_100i_0_7089_66.png)
Incorrect detection
![Incorrect detection](./docs/images/kshap/kshap_2000s_100i_0_7189_66.png)

#### Pixel flipping

![Pixel flipping for Kernel Shap with 2000 iterations in 100 images](./docs/images/kshap/pixel_flipping_kshap_2000s_100i.png)

#### Localization metrics:
 - Positive attribution inside box (against all positive attribution): 23.6%
 - Box coverage:  64.8%
 - IoU: 20%
