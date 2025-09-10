# Student Performance Prediction Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-yellowgreen)
![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-lightgrey)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)

A comprehensive machine learning project that predicts student exam scores based on various academic and lifestyle factors using regression techniques.

## 📊 Project Overview

This project analyzes the relationship between student characteristics and their academic performance. The goal is to build a predictive model that can estimate exam scores based on factors like study hours, attendance, parental involvement, and other relevant features.

## 🎯 Key Features

- **Data Cleaning & Preprocessing**: Handled missing values and encoded categorical variables using advanced strategies
- **Exploratory Data Analysis**: Interactive visualizations and correlation analysis to understand data relationships
- **Multiple Modeling Approaches**: Compared Linear Regression with Polynomial Regression
- **Comprehensive Evaluation**: Multiple performance metrics and visualization of results
- **Interactive Dashboard**: Built with Plotly and ipywidgets for data exploration

## 📁 Dataset

**Source**: Student Performance Factors (Kaggle)  
**Size**: 6,607 students × 20 features  
**Target Variable**: `Exam_Score` (numerical)

**Key Features Include**:
- `Hours_Studied`: Daily study hours
- `Previous_Scores`: Historical academic performance
- `Attendance`: Class attendance percentage
- `Teacher_Quality`: Categorical (Low, Medium, High)
- `Internet_Access`: Binary (Yes/No)
- `Sleep_Hours`: Average nightly sleep
- `Extracurricular_Activities`: Binary (Yes/No)
- And 13+ additional features...

## 🛠️ Technical Implementation

### Data Preprocessing
- **Missing Values**: Implemented "Missing" category strategy for categorical variables
- **Feature Encoding**: Advanced mixed encoding approach:
  - Label Encoding for binary features (Yes/No → 1/0)
  - One-Hot Encoding for multi-category features
- **Train-Test Split**: 80-20 split with random state for reproducibility

### Modeling Techniques
1. **Linear Regression**: Baseline model
2. **Polynomial Regression**: Degree 2 & 3 for capturing non-linear relationships

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

## 📈 Results

### Best Performing Model
- **Algorithm**: Linear Regression (Degree 1)
- **R² Score**: 0.7699
- **MAE**: 0.45 points
- **RMSE**: 1.80 points

### Key Insights
- Previous academic scores are the strongest predictor of exam performance
- Teacher quality shows significant impact on student outcomes
- Study hours have a non-linear relationship with exam scores (diminishing returns)
- The model explains 89% of variance in student exam scores

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.comMohamedElkhateb05/Student-Performance-Predictor-.git
   cd student-performance-predictor-
