# Dream-Dual-Memory-AD

This repository contains the code for the Master's thesis: **"Detection and localisation of real-world industrial anomalies using diffusion-based outliers synthesis"**.

The method synthesizes realistic anomalies using a diffusion model (Stable Diffusion) adapted to the industrial domain, and then uses these synthetic outliers to train a discriminative model (PatchCore) for anomaly detection and localization.

## 📋 Project Structure

- `src/scripts/`: Main execution scripts for the 4-step pipeline.
- `src/trainers/`: Trainer classes for Dreambooth and MVTec AD pretraining.
- `src/patchcore/`: Implementation of the PatchCore algorithm.
- `configs/`: Configuration files for Stable Diffusion.
- `dream-environment.yml`: Conda environment for the diffusion synthesis part.
- `patchcore-environment.yml`: Conda environment for the anomaly localization part.

## 🛠️ Installation

This project requires **two separate environments** to avoid dependency conflicts between the diffusion model (older PyTorch/CUDA) and the PatchCore implementation.

### 1. Dreamood Environment (Steps 1-4)
Used for training the diffusion model and generating synthetic anomalies.

```bash
conda env create -f dream-environment.yml
conda activate dreamood
```

### 2. PatchCore Environment (Step 5)
Used for the final anomaly detection and localization.

```bash
conda env create -f patchcore-environment.yml
conda activate patchcore-env
```

## 📂 Data Preparation

1.  **MVTec AD Dataset**: Download and extract the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).
2.  **ImageNet**: (Optional/Required for pretraining) Download ImageNet validation set or relevant classes.
3.  **DTD (Describable Textures Dataset)**: Download [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) for texture regularization during Dreambooth training.

Suggested structure:
```
data/
├── mvtec_ad/
│   ├── hazelnut/
│   ├── screw/
│   └── ...
├── dtd/
│   └── images/
└── imagenet/
```

## 🚀 Usage

The pipeline consists of 4 main steps (plus an anomaly localization step).

### Step 1: Domain Adaptation with Dreambooth
Fine-tune Stable Diffusion on normal images from the MVTec AD class to learn the specific object appearance.

**Script:** `src/scripts/train_dreambooth_ldm.py`
**Environment:** `dreamood`

```bash
python src/scripts/train_dreambooth_ldm.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --instance_data_dir "data/mvtec_ad/hazelnut/train/good" \
  --class_data_dir "output/hazelnut/class_images" \
  --output_dir "output/hazelnut/dreambooth" \
  --instance_prompt "a photo of a hazelnut" \
  --class_prompt "a photo of a object" \
  --dtd_root_dir "data/dtd/images" \
  --category "hazelnut" \
  --output_weights_path "output/hazelnut/weights/model.ckpt" \
  --output_text_encoder_path "output/hazelnut/weights/text_encoder.pth" \
  --with_prior_preservation --prior_loss_weight 1.0 \
  --resolution 512
```

### Step 2: Learning Text-Conditioned Visual Space
Train a ResNet model to align visual features with text embeddings, creating a "Visual Memory" (anchor).

**Script:** `src/scripts/pretrain_ad_mvtec_imagenet.py`
**Environment:** `dreamood`

First, create a JSON file (e.g., `text_encoders.json`) matching prompts to the text encoder weights from Step 1:
```json
{
  "a photo of a hazelnut": "output/hazelnut/weights/text_encoder.pth"
}
```

Then run:
```bash
python src/scripts/pretrain_ad_mvtec_imagenet.py \
  --mvtec-root "data/mvtec_ad" \
  --mvtec2-root "data/mvtec_ad" \
  --imagenet-root "data/imagenet" \
  --text_encoders_file "text_encoders.json" \
  --output-dir "output/embeddings" \
  --stats-file "stats.json" \  # Create empty or provide stats if needed
  --epochs 10
```
This generates `output/embeddings/anchor.npy` and `output/embeddings/embeds/{category}_embeddings.npy`.

### Step 3: Sample Outlier Embeddings
Generate latent embeddings that represent anomalies by sampling from the learned visual space.

**Script:** `src/scripts/get_embeddings.py`
**Environment:** `dreamood`

```bash
python src/scripts/get_embeddings.py \
  --classname "hazelnut" \
  --anchor-file "output/embeddings/anchor.npy" \
  --embed-file "output/embeddings/embeds/hazelnut_embeddings.npy" \
  --save_dir "output/hazelnut/outlier_embeddings" \
  --gaussian_mag 0.07 \
  --K_in_knn 200 \
  --gpu 1
```

### Step 4: Generate Outliers (Dreamood)
Synthesize images containing anomalies using the fine-tuned model and the sampled outlier embeddings.

**Script:** `src/scripts/dreamood_inpaint.py`
**Environment:** `dreamood`

```bash
python src/scripts/dreamood_inpaint.py \
  --ckpt "output/hazelnut/weights/model.ckpt" \
  --loaded_embeddings "output/hazelnut/outlier_embeddings/outlier_npos_embed_noise_0.07_K_200_select_None.npy" \
  --data_dir "data/mvtec_ad" \
  --output_dir "output/hazelnut/synthetic_anomalies" \
  --anomaly_class "hazelnut" \
  --prompt "a photo of a hazelnut" \
  --num_samples 1 --mask_generation_ratio 0.1 0.2
```

### Step 5: Anomaly Localization (PatchCore)
Run PatchCore using the real normal data and the generated synthetic anomalies.

**Script:** `src/scripts/run_patchcore.py`
**Environment:** `patchcore-env`

For MVTec AD 2 (or when submission generation is needed), use the `--save_private_submission` flag to process the private test set and save anomaly maps for submission.

```bash
# Example for MVTec AD 2 (e.g., class 'can')
export PYTHONPATH=.
python src/scripts/run_patchcore.py \
  --gpu 0 --num_seeds 1 --save_segmentation_images \
  --log_group vit_1.0 --log_project MVTecAD_Results \
  --all_results_csv_file "patchcore_results.csv" \
  --save_private_submission --submission_dir "output/submission" \
  --num_stds 3.0 \
  "output/patchcore_results" \
  patch_core \
    -b vit_large_dino \
    -le blocks.5 -le blocks.7 -le blocks.11 -le blocks.14 \
    -le blocks.17 -le blocks.20 -le blocks.22 \
    --alpha 10.0 \
    --metric_type IP --anomaly_score_fn ratio --faiss_on_gpu --score mean \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 25 --patchsize 3 -alt 85.0 -aut 100.0 \
  sampler -p 0.1 approx_greedy_coreset_cosine \
  dataset \
    --resize 518 --imagesize 518 --num_workers 4 \
    --apply_augmentations \
    --synth_anomalies_path "output/hazelnut/synthetic_anomalies" \
    --dataset-type mvtec2 \
    -d "can" \
    mvtec2 "data/mvtec_ad"
```

## 🖼️ Generated Examples

Here are some examples of synthetic anomalies generated by the pipeline:

| Transistor | Screw | Capsule |
| :---: | :---: | :---: |
| ![Transistor](synthetic_anomalies_example/transistor/sample_00003.png) | ![Screw](synthetic_anomalies_example/screw/sample_00011.png) | ![Capsule](synthetic_anomalies_example/capsule/sample_00123.png) |

| Can | Vial | Fabric |
| :---: | :---: | :---: |
| ![Can](synthetic_anomalies_example/can/sample_00007.png) | ![Vial](synthetic_anomalies_example/vial/sample_00016.png) | ![Fabric](synthetic_anomalies_example/fabric/sample_00023.png) |

