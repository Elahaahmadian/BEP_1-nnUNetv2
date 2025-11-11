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




