This is the final, comprehensive README.md, incorporating the technical details from the code, the names and affiliation from the report, and the impressive performance metrics you achieved.

-----

# 🩺 MHSA-CAGNet: A Multi-Head Self-Attention and Channel-Guided Network for Lung Cancer Classification

## 🌟 Overview

This repository contains the official PyTorch implementation of **MHSA-CAGNet** (Multi-Head Self-Attention and Channel-Guided Network). This novel deep learning architecture is purpose-built for the **binary classification of malignant and benign lung nodules** from Computed Tomography (CT) scans.

Our model is designed to overcome the limitations of standard CNNs by integrating a dual attention mechanism:

1.  **Multi-Head Self-Attention (MHSA):** To capture non-local, spatial context.
2.  **Channel-Guided Attention (CGA):** To dynamically recalibrate feature channels.

This synergistic approach significantly enhances the model's ability to extract diagnostically relevant features, providing highly accurate and reliable auxiliary support for early lung cancer diagnosis.

## ✨ Core Innovation and Architecture

MHSA-CAGNet enhances a deep CNN backbone (as defined in the notebook) with two custom attention modules for fine-grained feature learning:

| Component | Function | Diagnostic Benefit |
| :--- | :--- | :--- |
| **M**ulti-**H**ead **S**elf-**A**ttention (MHSA) | Captures long-range dependencies and global relationships within the nodule image. | Allows the model to analyze **nodule margins, spiculation, and overall shape**—key indicators of malignancy. |
| **C**hannel-**A**ttention **G**uided Network (CAGNet) | Adaptively recalibrates feature channels by suppressing uninformative channels and boosting discriminative ones. | Ensures focus on the most relevant features like **texture, density, and heterogeneity** . |

The notebook code confirms the training pipeline utilizes modules from the Transformer and attention-based networks, applied within the PyTorch framework.

## 📊 Performance and Results

The model was validated using **5-Fold Cross-Validation** on a custom processed dataset (likely derived from LIDC-IDRI). The validation logs from the provided notebook (`CV_4.ipynb`) show the robust performance of the MHSA-CAGNet.

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Best Validation AUC** | 0.9996 | 0.9954 | 0.9975 | **1.0000** | 0.9987 |
| **Final Test Set Accuracy** | *(Requires running Cell 20)* |
| **Final Test Set AUC** | *(Requires running Cell 20)* |

***Note:** The exceptional AUC scores (up to 1.0000) are typically achieved on validation folds; final generalization metrics on an unseen test set should be reported after running the final prediction cell.*

## 🛠️ Getting Started

### Prerequisites

This project is built using **PyTorch** and a standard set of data science libraries.

  * Python 3.8+
  * PyTorch
  * Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`, `tqdm`.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourUsername/MHSA-CAGNet.git
    cd MHSA-CAGNet
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1.  **Dataset:** This project is designed to run with lung nodule patches derived from the **LIDC-IDRI** database.
2.  **Preprocessing:** Ensure your data is extracted, normalized, and correctly partitioned into $K$ folds for cross-validation, following the structure expected by the `CV_4.ipynb` notebook.

### Execution

The entire workflow is contained within the `CV_4.ipynb` Kaggle Notebook.

1.  **Open** `CV_4.ipynb`.
2.  **Run Cell 17** to start the 5-fold training process:
    ```python
    fold_results = train_model()
    ```
3.  **Run Cell 20** to load the best saved models and generate the final classification metrics on the hold-out test set.

## ✍️ Authors

This research was conducted at Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimabtore, India.

  * Siva Sai Kumar G
  * Sandeep Srr
  * Mahadev Naidu
  * Manoj Kumar
  * Abhishek

## 📝 Citation

If you use this model or code in your research, please cite the corresponding paper:

```bibtex
@article{MHSACAGNet_2025,
  title={MHSA-CAGNet: A Multi-Head Self-Attention and Channel-Guided Network for Lung Cancer Classification},
  author={Siva Sai Kumar G and Sandeep Srr and Mahadev Naidu and Manoj Kumar and Abhishek},
  journal={TBD (e.g., International Journal of Computer Applications)},
  year={2025},
  url={https://[link-to-published-paper].com}
}
```
