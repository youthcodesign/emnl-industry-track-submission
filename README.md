# emnl-industry-track
This is the anonymous github repo for EMNLP industry track submission
---

## 📂 Directory Structure and Purpose

- **`data/`**
  - **`GOLD_TEST.csv`**
     BullyBench --> GOLD_TEST  
  - **`all_data/role_all/`**  
    Used as input by `train_test.py`. See the `src/` directory for training and testing scripts.
  
  - **`intrinsic/`**  
    Contains the annotation data used by `human_eval_final.ipynb` for intrinsic evaluation.

- **`supplementary_files/`**
  - Contains supporting files such as **annotation guides (PDFs)** and other documentation to assist in understanding and reproducing the evaluation process.

---

## Intrinsic Evaluation of Annotated Data

This repository contains resources and code used for intrinsic evaluation of annotated data across multiple user types and annotators.

---

### 📘 Main Notebook

- **`human_eval_final.ipynb`**  
  This notebook is designed to **calculate and analyze responses** from the **Intrinsic Evaluation** process for **five distinct annotators**.

---

### 🧑‍💻 Annotators and Roles

| User              | Annotator Type       |
|-------------------|----------------------|
| OnlineObserver    | Social Scientist     |
| SafetyEvaluator   | Social Scientist     |
| StreamerDreamer   | Content Moderator    |
| HideNShare        | Content Moderator    |
| CloudSafeZone     | Adult Teen           |

---

Feel free to reach out for any clarifications regarding the setup or usage.
