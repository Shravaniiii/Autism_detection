# Early Autism Spectrum Disorder (ASD) Detection using Machine Learning

A machine learning system for early detection of Autism Spectrum Disorder across two populations — **toddlers** and **adults** — using real-world behavioral screening datasets.

---

## Project Overview

Autism Spectrum Disorder (ASD) affects millions of people worldwide, yet early detection remains a challenge. This project builds a classification pipeline that uses behavioral screening scores and demographic features to predict ASD diagnosis, with the goal of supporting faster and more accessible early intervention.

The project analyzes two separate datasets:
- **Toddler Dataset** — behavioral Q-CHAT screening scores for children
- **Adult Dataset** — AQ-10 behavioral screening scores for adults

---

## Dataset

| Dataset | Source |
|---|---|
| Toddler Autism Dataset (July 2018) | UCI / Kaggle |
| Adult Autism Screening Dataset | UCI Machine Learning Repository |

**Key features used:**
- Behavioral screening scores (A1–A10)
- Age, Gender, Ethnicity
- Jaundice history
- ASD diagnosis label (target variable)

---

## Project Pipeline

### 1. Data Preprocessing
- Dropped null values from both datasets
- Standardized column names and labels
- Encoded categorical variables (gender, ethnicity, ASD label) numerically
- Applied `StandardScaler` for feature normalization

### 2. Exploratory Data Analysis (EDA)
- Ethnicity distribution (pie charts) for toddlers and adults
- ASD diagnosis breakdown by ethnicity and gender
- Jaundice history vs. ASD diagnosis
- Age distribution across both populations

### 3. Feature Selection
- Used **Random Forest feature importance** to identify the top 6 most predictive features for each population
- Selected features prioritized behavioral screening scores alongside demographic variables

### 4. Model Training & Evaluation

Trained and compared the following classifiers on both datasets:

| Model | Evaluation Metrics |
|---|---|
| Random Forest | Accuracy, Precision, Recall, F1-score |
| Support Vector Machine (SVM) | Accuracy, Precision, Recall, F1-score |
| Decision Tree | Accuracy, Precision, Recall, F1-score |
| K-Nearest Neighbors (KNN) | Accuracy, Precision, Recall, F1-score |

Evaluation included **confusion matrices** and full **classification reports** for each model.

> ✅ **SVM outperformed all other models** for both toddlers and adults.

### 5. Interactive Prediction System
Built a command-line prediction tool allowing users to input behavioral scores and demographic information to receive a real-time ASD prediction using the trained SVM model — for both toddlers and adults.

---

## Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn

---

## Results Summary

- SVM achieved the best classification performance across both datasets
- Behavioral screening scores (A-scores) were consistently the most predictive features
- Demographic features (ethnicity, age, gender) provided additional signal especially for toddlers

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Place the datasets in the working directory:
   - `Toddler Autism dataset July 2018.csv`
   - `autism_screening.csv`
4. Run the notebook:
   ```bash
   jupyter notebook Autism_Updated.ipynb
   ```

---

## Author

**Shravani Maskar**  
B.Tech in Computer Science (AI & ML), SRM Institute of Science and Technology  
MS in Data Science, Boston University  
[LinkedIn](https://www.linkedin.com/in/shravani-maskar/)
