# BEP_1-nnUNetv2

Segmentation of non-perfusion brain tissue on DSA images using nnU-Net v2 (2D).

## Overview
This project trains and evaluates nnU-Net v2 (2D) to segment non-perfused brain regions on pre-EVT AP miniIP DSA images. It covers preprocessing, 5-fold cross-validation, ensembling, postprocessing, metrics (Dice, HD95) and qualitative visualizations.

## Repository Structure
- nnUNet_raw_data_base – Original dataset (imagesTr, imagesTs, labelsTr, labelsTs)
- preprocessed_data
- model_results (folds 0–4)
- ensemble_predictions_final
- postprocessed_predictions_final
- Evaluation
## Environment (TU/e HPC)
```bash
module load cuda12.6/toolkit
conda activate nnunet_py310
export nnUNet_raw="/home/20203964/BEP_1/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/20203964/BEP_1/preprocessed_data"
export nnUNet_results="/home/20203964/BEP_1/model_results"
```

## 1) Preprocess
```bash
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity -np 1
```
## 2) Training ( 5-fold crossvalidation)
```bash
nnUNetv2_train 100 2d 0
nnUNetv2_train 100 2d 1
nnUNetv2_train 100 2d 2
nnUNetv2_train 100 2d 3
nnUNetv2_train 100 2d 4
```
## 3) Inference (test set)
```bash
nnUNetv2_predict \
  -d Dataset100_nonperfusion \
  -i /home/20203964/BEP_1/nnUNet_raw_data_base/Dataset100_nonperfusion/imagesTs \
  -o /home/20203964/BEP_1/ensemble_predictions_final \
  -f 0 1 2 3 4 \
  -tr nnUNetTrainer \
  -c 2d \
  -p nnUNetPlans
```

## 4) Postprocessing
```bash
nnUNetv2_apply_postprocessing \
  -i /home/20203964/BEP_1/ensemble_predictions_final \
  -o /home/20203964/BEP_1/postprocessed_predictions_final \
  -pp_pkl_file /home/20203964/BEP_1/model_results/Dataset100_nonperfusion/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
  -plans_json /home/20203964/BEP_1/model_results/Dataset100_nonperfusion/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json \
  -np 4
```

## 5) Evaluation
```bash
GT="/home/20203964/BEP_1/nnUNet_raw_data_base/Dataset100_nonperfusion/labelsTs"
PRED="/home/20203964/BEP_1/ensemble_predictions_final"
DJ="/home/20203964/BEP_1/model_results/Dataset100_nonperfusion/nnUNetTrainer__nnUNetPlans__2d/dataset.json"
PL="/home/20203964/BEP_1/model_results/Dataset100_nonperfusion/nnUNetTrainer__nnUNetPlans__2d/plans.json"
MET_TXT="$PRED/metrics.txt"
MET_JSON="$PRED/metrics.json"

nnUNetv2_evaluate_folder \
  "$GT" \
  "$PRED" \
  -djfile "$DJ" \
  -pfile "$PL" \
  -np 1 \
  -o "$MET_JSON" \
  2>&1 | tee "$MET_TXT"
```
### HD95

```python
import os
import SimpleITK as sitk
from medpy.metric.binary import hd95

pred_dir = "/home/20203964/BEP_1/labelspost_predicted_2D"
gt_dir = "/home/20203964/BEP_1/labelsTs_2D"

pred_files = sorted(os.listdir(pred_dir))
gt_files = sorted(os.listdir(gt_dir))

for pred_file, gt_file in zip(pred_files, gt_files):
    pred_path = os.path.join(pred_dir, pred_file)
    gt_path = os.path.join(gt_dir, gt_file)

    pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

    try:
        # HD95 berekenen
        hd = hd95(gt_img, pred_img)
        print(f"{pred_file} vs {gt_file}: HD95 = {hd:.2f} mm")
    except Exception as e:
        print(f"Fout bij {pred_file} vs {gt_file}: {e}")

```

### Dependencies

This project was executed on the TU/e High Performance Computing (HPC) cluster.

**Core environment:**
- Python 3.10
- CUDA 12.6
- nnU-Net v2.2
- PyTorch 2.3+
- NiBabel
- SimpleITK
- Pillow
- Matplotlib
- NumPy 
- seg-metrics 
- medpy

### Example SLURM job 
```bash
#!/bin/bash
#SBATCH --job-name=BEP1_Train
#SBATCH --partition=bme.gpu2.q
#SBATCH -o logs/evaluation_%j.out
#SBATCH -e logs/evaluation_%j.err
#SBATCH --time=25:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
```

## Additional Scripts

Convert NIfTI predictions to 2D slices
This script converts 3D NIfTI files (e.g., `[1, H, W]` or `[D, H, W]`) into 2D NIfTI files for visualization or further processing


```python
import os
import SimpleITK as sitk

input_dir = "/path/to/imagesTr"  # or imagesTs, labelsTr, labelsTs
output_dir = "/path/to/output_2D"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".nii.gz"):
        img = sitk.ReadImage(os.path.join(input_dir, fname))
        arr = sitk.GetArrayFromImage(img)

        # Take middle slice if 3D
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr_2d = arr[arr.shape[0] // 2]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr_2d = arr[0]
        else:
            arr_2d = arr

        fixed_img = sitk.GetImageFromArray(arr_2d)
        fixed_img.SetSpacing((1.0, 1.0))
        sitk.WriteImage(fixed_img, os.path.join(output_dir, fname))
```

# Automatic Segmentation of Non-Perfused Brain Tissue in DSA Images Using nnU-Net v2

This project focuses on the development and evaluation of an automatic segmentation model to identify non-perfused brain tissue in Digital Subtraction Angiography (DSA) images of acute ischemic stroke patients. The goal of the segmentation is to support treatment assessment and potentially help clinicians evaluate the success of endovascular thrombectomy (EVT).

The dataset used in this project consists of pre-EVT anterior–posterior (AP) projection DSA images. Manual segmentations were created to serve as ground truth labels. The nnU-Net v2 framework was used to train and evaluate a 2D segmentation model.

---

## Repository Structure

| Folder | Description |
|-------|-------------|
| **nnUNet_raw_data_base** | Original dataset structured according to nnU-Net: training and test images with corresponding labels. |
| **preprocessed_data** | Output of the preprocessing stage performed by nnU-Net (normalization, resampling, cropping). |
| **model_results** | Contains trained model checkpoints and training logs for all five cross-validation folds. |
| **ensemble_predictions_final** | Raw ensemble predictions created by combining the outputs of all trained folds. |
| **postprocessed_predictions_final** | Final cleaned segmentation masks after applying nnU-Net’s post-processing pipeline. |
| **Evaluation** | Contains evaluation outputs, including Dice scores and Hausdorff95 distance results. |

---

## Computational Environment (TU/e HPC)

This project was executed entirely on the TU/e High Performance Computing (HPC) cluster. The environment consisted of Python 3.10, CUDA-enabled GPU acceleration, PyTorch, and nnU-Net v2. All required environment variables were set so that nnU-Net could locate the raw dataset, preprocessed dataset, and model results directories.

---

## Method Overview

### 1. Data Preparation  
The dataset was organized into nnU-Net’s expected format and verified for consistency. nnU-Net automatically handled normalization, resampling, cropping, and patch configuration during preprocessing.

### 2. Model Training (5-Fold Cross Validation)  
The model was trained using the 2D nnU-Net framework. Five folds were trained independently, ensuring that every case contributed to both training and validation. This helps evaluate the robustness and generalization capability of the model.

### 3. Ensemble Prediction  
For testing, segmentations from all five trained models were combined (ensembled). Ensemble averaging typically produces more stable and reliable segmentation outputs than using a single model.

### 4. Post-Processing  
The ensemble output was refined by removing isolated false-positive regions and ensuring spatial consistency. This step significantly improved boundary accuracy and reduced noise in the predictions.

### 5. Evaluation  
Performance was assessed using Dice similarity coefficient and Hausdorff95 distance.  
- **Dice** measures overlap between prediction and ground truth.  
- **Hausdorff95 (HD95)** measures contour / boundary accuracy.

Post-processing improved segmentation smoothness and reduced extreme boundary errors.

---

## Key Dependencies

This project relies on:
- nnU-Net v2 framework
- PyTorch for GPU-accelerated training
- SimpleITK and NiBabel for medical image handling
- seg-metrics or medpy for quantitative evaluation

---

## Additional Notes for Future Work

- The current model is trained only on **AP projection** DSA images. Including lateral or oblique projections may improve generalization.
- The ground truth segmentations were created manually, and small annotation inconsistencies can influence evaluation metrics.
- Expanding the dataset or incorporating clinical validation could strengthen the clinical relevance of this approach.

---

## Purpose of This Repository

This repository is structured to allow future students to:
- Continue model development
- Reproduce evaluation experiments
- Train new models on extended datasets
- Experiment with 3D nnU-Net or hybrid projection inputs

All required datasets, model outputs, and processing pipelines are included so the project can be resumed **without needing to repeat preprocessing or data setup.**


