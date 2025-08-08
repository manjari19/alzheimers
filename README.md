# Alzheimer's Detectives  
**Early Detection of Alzheimer’s Using Clinical Data**  
CMPT 310 – D200: Introduction to Artificial Intelligence and Machine Learning (Summer 2025)

**Team Members:**  
- Beyzanur Kuyuk  
- Manjari Prasad  
- Wan Yu Wendy Wong  

---

## Project Overview

This project applies supervised machine learning to enable **early detection of Alzheimer’s Disease** using clinical and cognitive data from the **OASIS Cross-Sectional Dataset**.  

We built a **binary classification system** that predicts whether an individual is at risk for dementia using features such as:
- Age  
- MMSE score  
- Education level  
- Socioeconomic status (SES)  
- MRI biomarkers  
- Gender  

Our primary goals:
- Aid in **early diagnosis**  
- Provide **transparent and interpretable results**  
- Build an **accessible clinical decision support tool**

---

##  Key Methods & Technologies

###  Languages & Libraries
- Python  
- pandas  
- scikit-learn  
- seaborn, matplotlib  
- imbalanced-learn  

###  Dataset
- **OASIS Cross-Sectional Dataset** from Kaggle  
  [https://www.kaggle.com/code/vaibhavmathur96/detecting-early-alzheimer-s](https://www.kaggle.com/code/vaibhavmathur96/detecting-early-alzheimer-s)

### Models Used
- Logistic Regression (with and without class weighting)  
- Random Forest  

### Techniques
- 10-Fold Cross-Validation  
- Class Weighting (`class_weight='balanced'`)  
- RandomOverSampler (to address class imbalance)

---

## AI Pipeline

1. **Raw Data:** Demographics, MMSE, SES, CDR, MRI indicators  
2. **Data Cleaning:**  
   - Drop rows with missing CDR  
   - Impute missing MMSE, SES, Educ  
   - Encode gender  
3. **Preprocessing:**  
   - Scale numeric features  
   - Create binary label (CDR > 0)  
4. **Training:**  
   - Logistic Regression & Random Forest  
   - 10-Fold Cross-Validation  
5. **Evaluation:**  
   - Accuracy, F1 Score, Precision, Recall  
   - Confusion Matrix  
   - ROC Curve  

---

## Summary of Results

| Metric     | Logistic Regression | Random Forest |
|------------|---------------------|----------------|
| Accuracy   | 80%                 | 83%            |
| F1 Score   | 0.80                | 0.805          |
| Precision  | 0.73                | 0.82           |
| Recall     | 0.90                | 0.79           |

---

## Key Insights

- **Logistic Regression** performed best for early detection due to higher **recall (0.90)**.  
- **Random Forest** had **higher precision (0.82)**, reducing false positives.  
- **MMSE score** was the **strongest predictor** of dementia.  
- Class weighting and oversampling both improved performance.  
- Cross-validation ensured robust model evaluation.  

---

## How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/manjari19/alzheimers.git
   cd alzheimers
   ```

2. **Create virtual environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
  

4. **Run preprocessing and model training**
   ```bash
   python preprocessing.py
   python logistic_model.py
   python randomforest.py
   ```

5. **Optional (for advanced training & validation)**
   ```bash
   python logistic_model_oversampling.py
   python Logistic_model_cross_val.py
   python random_forest_crossval.py
   ```

6. **Evaluate & visualize results**
   ```bash
   python Logistic_model_cross_val.py
   python visualizations.py
   ```
 

---

##  References

- Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007).  
  Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults. *Journal of Cognitive Neuroscience, 19*(9), 1498–1507.  
  https://doi.org/10.1162/jocn.2007.19.9.1498

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011).  
  Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

- He, H., & Garcia, E. A. (2009).  
  Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.  
  https://doi.org/10.1109/TKDE.2008.239

---

## Acknowledgements

We acknowledge the contributions of the CMPT 310 teaching team.  
Dataset courtesy of OASIS (Marcus et al.).  
Poster template adapted from Canva.

---