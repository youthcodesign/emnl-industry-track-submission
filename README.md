# emnl-industry-track
This is the anonymous github repo for EMNLP industry track submission
---

## Hyper-parameter configurations for **BASELINE** on  𝓓<sub>Gold</sub>

### RoBERTa-base & GPT2-medium Top-2 on 𝓓<sub>Gold</sub>

| Model       | Train-Eval Data                       | Strategy   | Loss Function  | Upsampling? | Fold | Macro-F1 | Enabler F1 | Defender F1 | Bully F1 | Victim F1 |
|-------------|----------------------------------------|------------|----------------|-------------|------|----------|-------------|--------------|-----------|------------|
| RoBERTa     | **𝓓₁ ∪ 𝓓₂ ∪ 𝓓₃**                        | Pair-wise  | Focal Loss     | True        | 2    | 0.4034   | 0.4422      | 0.6052       | 0.3420    | 0.2243     |
| RoBERTa     | **𝓓₁ ∪ 𝓓₂ ∪ 𝓓₃**                        | Multiclass | Cross Entropy  | True        | 4    | 0.3911   | 0.5156      | 0.5700       | 0.3085    | 0.1702     |
| GPT2-medium | **𝓓₁ ∪ 𝓓₂ ∪ 𝓓₃**                        | Multiclass | Focal Loss     | True        | 4    | 0.3807   | 0.5234      | 0.5775       | 0.2317    | 0.1905     |
| GPT2-medium | **𝓓₁ ∪ 𝓓₂ ∪ 𝓓₃**                        | Pairwise   | Cross Entropy  | True        | 1    | 0.3756   | 0.5000      | 0.5519       | 0.2919    | 0.1587     |

### Configuration files




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
