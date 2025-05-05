# Human-Activity-Recognition-with-Smartphones
Leverage hardware data from smart devices to recognise and categorise activity into six categories: standing, sitting, laying, walking, walking upstairs, and walking downstairs using machine learning.

## Project Overview

This project aims to predict and classify human activities using sensor data collected from smartphones. Using machine learning techniques, especially XGBoost, we developed a robust classifier for real-time human activity recognition based on a publicly available dataset.

## Dataset

- **Source:** UCI Human Activity Recognition (HAR) Dataset
- **Participants:** 30 individuals
- **Device:** Waist-mounted smartphones
- **Activities Tracked:**  
  - Walking  
  - Walking Upstairs  
  - Walking Downstairs  
  - Sitting  
  - Standing  
  - Laying  
- **Features:** 561 time and frequency domain variables collected from sensors  
- **Data Quality:** No missing values, balanced classes in target column

## Problem Statement

Develop a machine learning model to accurately classify a person’s physical activity based on smartphone sensor data.

## Exploratory Data Analysis

- Box plots show high variability in activity data.
- Scaling and normalization are essential.
- No class imbalance or missing data issues.

## Preprocessing

- **Standard Scaler** used for normalization.
- **Label Encoder** applied to convert categorical labels.
- **PCA** (in Approach 2) retained 90% variance using 63 principal components.

## Models Used

- Random Forest
- SVM
- KNN
- Decision Tree
- Naive Bayes
- Gradient Boosting
- AdaBoost
- **XGBoost** (best performing)

## Modeling Approaches

### Approach 1: Using Raw Data
- **Training Time:** 8 minutes
- **XGBoost Accuracy:**
  - Train: 100%
  - Validation: 99%
  - Test: 93%
- **Observation:** Overfitting observed

### Approach 2: Using PCA
- **Training Time:** 1 minute (87% faster)
- **XGBoost Accuracy:** Test: 89%
- **Observation:** More generalizable model with slight dip in accuracy

## Hyperparameter Tuning (XGBoost)

- **n_estimators:** 300  
- **max_depth:** 3  
- **learning_rate:** 0.2  
- **min_child_weight:** 1  

Tuning done using `GridSearchCV` resulted in improved accuracy from 89% to **90%**.

## Final Results

- XGBoost with PCA gave the **best balance of accuracy and training time**.
- **Challenges:** Confusion between similar activities like Sitting vs Standing.
- **Successes:** High overall accuracy and efficiency using standard ML models.

## Future Scope

- Explore L1/L2 regularization
- Precision-Recall optimization
- Transition to deep learning models (e.g., LSTM or CNN for time-series)
