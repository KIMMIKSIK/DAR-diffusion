# DAR-diffusion
Official Repository for "**Diffusion-Based Domain-Anchored Refinement for Vehicle Dent Detection Enhancement**"
This repository contains the data and code used for the experiments in our paper, which are based on the public dataset CarDD.

Before running the code, please install the required packages listed in `requirements.txt`.
All experiments were conducted with Python 3.10.16

## Dataset
Our work does not directly use images as input. Instead, we crop the defect regions and use them for training and evaluation.
The related dataset can be downloaded via the following link.
[https://cardd-ustc.github.io/]

## Code

### 1st Training

For the initial dent-patch generation stage, we fine-tuned a text-to-image diffusion model with LoRA using cropped dent images.  
The training script `train_distribute.py` was adapted from the Hugging Face `diffusers` text-to-image example code:

https://github.com/huggingface/diffusers/tree/main/examples/text_to_image

We sincerely thank the Hugging Face team for making this implementation publicly available.

Before running the script:
- You must prepare your own **Hugging Face account and access token**
- If logging is enabled, you must also provide your own **Weights & Biases (`wandb`) API key**

### 1st Generation
the learned LoRA weights are used to generate diverse initial dent patches using a Stable Diffusion pipeline (runwayml/stable-diffusion-v1-4, v1-5).

### Representative Sample Selection
The representative-sample selection script is `5_coreset_selection.py`.

Example usage:

```bash
python 5_coreset_selection.py \
  --images_dir path_to_damage_images \
  --labels_dir path_to_damage_labels \
  --output_dir path_to_output_dir
```

### DreamBooth Training
The DreamBooth training script is `train_dreambooth.py`.

Example usage:

```bash
python train_dreambooth.py --save_dir "path\to\save_model"
```

Before running the script, users should update image_dir inside the file.

The image_dir should contain the five representative images selected by 5_coreset_selection.py.
These five images are used as the few-shot samples for DreamBooth training.


### 2nd Refinement
The refinement script is `train_refinement.py`.

Example usage:

```bash
python train_refinement.py \
  --model_dir ./dreambooth_5test \
  --input_dir path_to_input_images \
  --output_dir path_to_output_images \
  --reference_crop_dir path_to_reference_dent_crops
```


### 3rd Overlay

The `overlay.py` script composites the refined dent images onto clean vehicle background images and automatically generates YOLO-format labels.

Example usage:

```bash
python overlay.py \
  --damage_images_dir path_to_background_images \
  --damage_labels_dir path_to_background_labels \
  --damage_ref_folder path_to_refined_dent_images \
  --new_damage_img_dir path_to_output_images \
  --new_damage_label_dir path_to_output_labels \
  --visual_output_dir path_to_visualization_images
```

```bash
python boundary_inpaint.py \
  --image_dir path_to_overlay_images \
  --label_dir path_to_overlay_labels \
  --save_dir path_to_inpainted_images \
  --label_output_dir path_to_inpainted_labels \
  --model_path path_to_inpainting_model \
  --lora_path path_to_lora_weights
```

## Citation
If you find our work helpful, please consider citing the following paper and ⭐ the repo.

```

```
