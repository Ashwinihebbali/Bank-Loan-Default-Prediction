# Bank Loan Prediction using Logistic Regression

## Overview

This project implements a **Logistic Regression model** to predict loan default status based on borrower and loan characteristics. The model is trained on historical bank loan data and can be used to classify new loan applications as likely to default or not default, helping financial institutions make informed lending decisions.

## Project Description

Bank loan default prediction is a critical business problem in credit risk management. This project develops a classification model that:
- Analyzes customer and loan features
- Identifies patterns associated with loan defaults
- Predicts the default probability for new loan applicants
- Provides useful insights into default risk factors

## Dataset

**File:** `bankloan.sas7bdat` (SAS7BDAT format)

The dataset contains bank loan records with the following characteristics:
- **Features:** Various borrower demographics, loan characteristics, and financial indicators
- **Target Variable:** `default` (binary: 0 = No Default, 1 = Default)
- **Data Processing:** 
  - Missing values in the target variable are separated and used for final predictions
  - Complete cases are used for model training and evaluation
  - Outliers are handled using the Interquartile Range (IQR) method

## Project Structure

```
Bank Loan Prediction/
├── README.md                        # Project documentation
├── Logistic Regression.ipynb        # Main analysis and modeling notebook
└── bankloan.sas7bdat               # Source data file (SAS format)
```

## Methodology

### 1. **Data Loading & Exploration**
   - Load data from SAS7BDAT format
   - Inspect data structure, types, and basic statistics
   - Identify missing values and data quality issues

### 2. **Data Preprocessing**
   - Separate records with missing target values for later prediction
   - Focus on complete cases for model development
   - Check for duplicates and unique values

### 3. **Outlier Detection & Treatment**
   - Apply IQR (Interquartile Range) method
   - Cap outliers at lower and upper fences: Q1 - 1.5×IQR and Q3 + 1.5×IQR
   - Visualize distributions before and after treatment

### 4. **Model Development**
   - **Train-Test Split:** 80% training, 20% testing (random_state=42)
   - **Algorithm:** Scikit-learn Logistic Regression
   - **Features:** All variables except the target

### 5. **Model Evaluation**
   - Training and testing accuracy
   - Classification Report (Precision, Recall, F1-Score)
   - Probability threshold optimization (tested thresholds: 0.5, 0.3)

### 6. **Predictions on New Data**
   - Predict loan default status for records with missing target
   - Generate probabilistic predictions
   - Apply optimized probability threshold (0.3) for final predictions

## Requirements

- **Python 3.7+**
- **Libraries:**
  - pandas
  - scikit-learn
  - matplotlib
  - numpy

## Installation

1. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn matplotlib numpy
   ```

2. **For SAS file support (optional):**
   ```bash
   pip install sas7bdat
   ```

## Usage

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook "Logistic Regression.ipynb"
   ```

2. **Run all cells in sequence** to:
   - Load and explore the data
   - Perform data cleaning and preprocessing
   - Train the logistic regression model
   - Generate evaluations and predictions

3. **Key Outputs:**
   - Model accuracy on training and test sets
   - Classification reports with precision, recall, and F1-scores
   - Predictions for records with missing default values
   - Probability estimates for each prediction

## Model Performance

The notebook includes evaluation metrics:

- **Training Accuracy:** Model performance on training data
- **Testing Accuracy:** Model generalization to unseen data
- **Classification Metrics:** Detailed performance by class (0 = No Default, 1 = Default)
  - Precision: Accuracy of positive predictions
  - Recall: Ability to identify positive cases
  - F1-Score: Harmonic mean of precision and recall

### Probability Threshold Optimization

The model uses predicted probabilities for classification:
- **Default threshold (0.5):** Balanced classification
- **Optimized threshold (0.3):** Adjusted based on business requirements and classification report analysis

## Features Used

The model uses bank loan features including:
- Education level (`ed`)
- Default status (target variable)
- Various customer and loan characteristics
- Numerical and categorical variables

## Key Insights

- Use the classification report to understand model trade-offs between precision and recall
- Boxplots visualize the distribution of features and help identify outliers
- The probability threshold can be adjusted based on business cost-benefit analysis
- The model successfully predicts default status on new loan applications

## Files Generated

During notebook execution:
- Visualizations (boxplots) of feature distributions
- Classification reports (training and test sets)
- Prediction data frames with probability scores
- Final predictions for new loan applicants

## Notes

- The notebook filters outliers using the IQR method for improved model stability
- Missing values in the target variable do not prevent analysis; they're held out for later prediction
- Random state is set to 42 for reproducibility
- Warnings are suppressed for cleaner output

## Future Enhancements

- Feature engineering and selection
- Hyperparameter tuning
- Cross-validation for robust evaluation
- Comparison with other classification algorithms
- Model deployment and API development
- ROC-AUC and other performance metrics
- SHAP values for feature importance analysis

## Author

Financial Machine Learning Project

## License

This project is for educational purposes.

---

**Last Updated:** February 2026

For questions or improvements, please refer to the notebook documentation and analysis details.
