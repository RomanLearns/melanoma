# Medical Image Classification for Early Melanoma Detection

This project is a Virginia Tech CS 4094 Capstone project for ML for Healthcare. The goal of the project is to classify dermoscopic skin lesion images as benign or malignant using transfer-learned convolutional neural networks. The project uses an ISIC-derived Kaggle dataset and compares several pretrained CNN architectures, including EfficientNet-B0, EfficientNet-B2, ResNet-50, DenseNet-121, and MobileNetV2.

The main workflow is contained in capstone_exploration.ipynb. This notebook includes dataset exploration, preprocessing, stratified train/validation splitting, augmentation examples, model training, model evaluation, and result visualization. The helper file cell_fixes.py contains supporting code used during the implementation process. The capstone_results/ directory contains saved benchmark results and figures used in the final report.

## Dataset

The dataset archive is stored as 002.zip. The dataset contains dermoscopic lesion images organized into benign and malignant classes. In the final project, the dataset was split using a stratified train/validation procedure to avoid class imbalance issues in validation. The held-out test set contains 2,000 images, with 1,000 benign and 1,000 malignant examples.

## How to Run the Project

1. Clone the repository.

2. Open the project folder.

3. Make sure the required Python packages are installed. The main packages used include:
   - Python
   - TensorFlow / Keras
   - NumPy
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn
   - Jupyter Notebook

4. Open capstone_exploration.ipynb in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.

5. Run the notebook cells in order. The notebook performs dataset loading, preprocessing, model training, evaluation, and figure generation.

6. Review the generated outputs in the capstone_results/ directory.

## Main Files

capstone_exploration.ipynb:
Main notebook for the project. It contains the data exploration, preprocessing, training, evaluation, and visualization workflow.

cell_fixes.py:
Supporting Python code used during implementation.

capstone_results/arch_benchmark.pkl:
Saved architecture benchmark results.

capstone_results/arch_benchmark.png:
Visualization comparing model performance across architectures.

capstone_results/augmentation_examples.png:
Examples of image augmentations used during training.

capstone_results/class_distribution.png:
Class distribution figure used in the final report.

capstone_results/sample_images.png:
Sample dermoscopic images from the dataset.

## Final Model Summary

The strongest overall model in the final benchmark was ResNet-50. It achieved an AUC-ROC of 0.9800, accuracy of 0.9185, F1 score of 0.9154, and sensitivity of 0.9850 at 80% specificity on the balanced held-out test set.

## Notes

This project is intended as an academic machine learning prototype. It should not be used as a clinical diagnostic tool. Any real medical use would require additional validation, fairness analysis, clinical review, and regulatory approval.

Curated by: Roman Chenoweth, Bryan Torres, Cian Teague
