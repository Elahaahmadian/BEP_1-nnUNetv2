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
## 2) Training
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

    # Beelden inladen
    pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

    try:
        # HD95 berekenen
        hd = hd95(gt_img, pred_img)
        print(f"{pred_file} vs {gt_file}: HD95 = {hd:.2f} mm")
    except Exception as e:
        print(f"Fout bij {pred_file} vs {gt_file}: {e}")


### 8) Dependencies

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
- NumPy < 2.0
- seg-metrics 
- medpy 
