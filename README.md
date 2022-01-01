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

## Running
### Kernel SHAP
**Example command**: Run Kernel SHAP for the first 100 COCO images, using 2000 samples for approximating SHAP and assign for each perturbed feature a value of 0.5 (light gray), and store results.
```bash
python scripts/kernelshap_ssd.py \
    --first-images=100 \
    --batch-size=16 \
    --shap-samples=2000 \
    --baseline-value=0.5 \
    --result-file="data/results/kshap/kshap_2000s_100i.hdf5"
```

## Results
### Kernel SHAP
- Kernel SHAP trains a linear explanation model to locally approximate the true model (using LIME framework)
- Exact runtime grows exponentially with the number of features, i.e. pixels.
- To constraint the runtime, we
  -  segment image into 256 SLIC superpixels, each corresponding to one feature
  -  train a surrogate model on 2000 samples (perturbations) per explanation (detected box)
- Evaluated on the first 100 images of COCO 2017 using SSD detector


#### Examples:
Correct detection
![Correct detection](./docs/images/kshap/kshap_2000s_100i_0_7089_66.png)
Incorrect detection
![Incorrect detection](./docs/images/kshap/kshap_2000s_100i_0_7189_66.png)

#### Pixel flipping
 - Area Under the Curve (trapezoidal rule): ~ 7.023 (unnormalized), or 0.0236 (normalized against 254 boxes) 
![Pixel flipping for Kernel Shap with 2000 iterations in 100 images](./docs/images/kshap/pixel_flipping_kshap_2000s_100i.png)

#### Localization metrics
 - Positive attribution inside box (against all positive attribution): 23.6%
 - Box coverage:  64.8%
 - IoU: 20.9%

#### Runtime
 - On Google Colab's Tesla P100
 - Batch size = 128
 - 100 images (254 boxes): 2 hours 6 minutes
 - Average time per explanation (i.e. box): ~30 seconds
