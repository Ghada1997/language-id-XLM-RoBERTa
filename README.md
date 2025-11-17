# Language Identification with fine tuning XLM-R and using SVM as a Baseline

This project implements **multilingual language identification** using the **WiLI-2018 dataset** by Thoma (2018).   
The code is structured for reproducibility and modularity, following a clean research pipeline. 

---

## 🚀 Approach

Our approach applies fine-tuning of a pretrained multilingual transformers for language identification. The selected model is **XLM-RoBERTa (XLM-R)** by Conneau et al. (2020) due to its robustness.
We then compare a **transformer-based model (XLM-RoBERTa)** against a **traditional SVM baseline** by Cortes and Vapnik (1995) (character n-gram TF–IDF).


- **Task:** Language identification across **235 balanced languages** (WiLI-2018 dataset).
- **Dataset:** WiLI-2018 (Wikipedia Language Identification)
- **Tech stack:** Python, PyTorch, Hugging Face Transformers, Optuna, Scikit-learn, Pandas, Matplotlib, Seaborn  
- **Models:**  
  - **XLM-RoBERTa (XLM-R)** (Conneau et al., 2020): fine-tuned for this task.  
  - **SVM baseline**: character n-gram TF–IDF, a robust traditional method.  
- **Why both?**  
  - XLM-R demonstrates the effectiveness of modern transformer-based LLM approach for better capturing context (which is the main idea of the project).
  - SVM provides a strong traditional baseline.   
- **Privacy and Reproducibility Notes:**  
  - Dataset splits can be fully reproduced (`seed=42`).  
  - By default, the notebook also loads cached splits and models from Google Drive for convenience but their respective files are kept private since they be reproduced from scratch with the provided code.
  - The repository includes the Optuna tuning pipeline and parameter schema. All **best_params.json** files contains placeholder keys only; exact tuned values are withheld to preserve research integrity.

---

## 📂 Project Structure

language-id/
├─ README.md # Project description and usage notes
├─ requirements.txt # Python dependencies
├─ notebooks/
│ └─ project_report.ipynb # End-to-end workflow: EDA, preprocessing, training, evaluation, commenting on performance
├─ src/
│ └─ language_id/
│   ├─ ___init___.py
│   ├─ utils.py # Utility functions (set_seed, save/load JSON)
│   ├─ data.py # Load WiLI, train/val/test splits, stratified subsets
│   ├─ preprocess.py # Minimal whitespace normalization; tokenization
│   ├─ model.py # Model factory for XLM-R / mBERT
│   ├─ metrics.py # computing Accuracy, macro/weighted F1, precision, recall
│   ├─ tune.py # Optuna hyperparameter tuning (on subsets)
│   └─ train.py # Final training & evaluation with best parameters
├─ outputs/ # Generated artifacts (gitignored)
│ └─ metrics/ # JSONs, learning curves, best_params.json, validation.json and evaluation.json
└─ .gitignore


---

## ⚙️ Preprocessing

- **Minimal preprocessing**:  
  WiLI-2018 dataset is already clean and free of any inconsistencies as mentioned by the authors.
  No aggressive text cleaning (no stopword removal, stemming, or punctuation stripping).
  Only **whitespace normalization** was applied.  
  
- **For XLM-R**:    
  XLM-R expects raw text with casing, punctuation, and diacritics intact.  

- **For the SVM baseline**:  
  Character n-gram TF–IDF vectors were built without stopword removal or punctuation stripping.  
  Modern Language ID benefits from diacritics and punctuation (e.g., inverted punctuation in Spanish, guillemets in French).  

- **Dataset balance**:  
  Short paragraphs were retained to maintain WiLI’s balanced class design.  

---

## 🔎 Validation and Hyperparameter Tuning

- **Splits**:  
  - Original WiLI: train / test.  
  - Validation: was generated based on 90/10 split of the train set (stratified by language) (90% for train and 10% for val).  

- **Optuna tuning**:  
  - This repo includes the Optuna pipeline and search ranges. Exact best hyperparameters and full tuning logs are withheld; please run **tune.py** to obtain your own best parameters.
  - Conducted on a stratified subset of **200 examples per language** (~20% of train+val) to account for computational cost.  
  - Preserved class balance across 235 languages.  
  - Reduced training time per trial.
  - Total of 20 Trials.  

- **Final models**:  
  - Retrained on the full train split with best parameters.  
  - Validated on the 10% validation split.  
  - Evaluated on the held-out test set.  

- **Baseline SVM**: tuned with the same procedure for fairness.  

---

## 📊 Evaluation & Results

| Model         | Validation Macro-F1 | Test Macro-F1 | Test Accuracy |
|---------------|---------------------|---------------|---------------|
| **XLM-R**     | ~0.971  lower       | ~0.970        | ~0.970        |
| **SVM TF-IDF**| ~0.995 (baseline)   | ~0.959 lower  | ~0.959 lower  |

- **Macro-F1** was the main optimization metric because the dataset is balanced and all languages are equally important.  
- Accuracy and macro-F1 are closely aligned, confirming consistent performance across classes.  
- **XLM-R** clearly outperforms the baseline on the test set, showing the strength of transformer-based models.
- Although the SVM baseline reached a higher macro-F1 on the validation split, its decline on the test set reveals weaker generalization.  

---

## ✅ Recommended Model

- The **fine-tuned XLM-R** is recommended:  
  - Achieved ≈97% macro-F1 and accuracy.  
  - Stable performance across validation and test.  
  - Strong generalization potential.  

- The baseline SVM is included for comparison and reproducibility.  

---

## 📚 References

- Cortes, C., Vapnik, V. Support-vector networks. *Mach Learn* 20, 273–297 (1995).
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (p. 8440). Association for Computational Linguistics.
- Thoma, M. (2018). The WiLI benchmark dataset for written language identification. *arXiv preprint arXiv:1801.07779*.
...
