# Predictive Maintenance Cost Reduction Project

## Overview

This project aims to implement a predictive maintenance model to reduce maintenance costs for a fleet of trucks. By predicting which trucks are likely to have defects in their air systems, we can optimize maintenance schedules and avoid unnecessary costs.

## Objectives

- Develop a predictive model to identify trucks likely to experience defects.
- Compare the costs of traditional maintenance method adopted and the costs after the model.
- Evaluate the model's performance using relevant metrics.
- Visualize the cost savings and model performance.

## Table of Contents

- [Installation](#installation)
- [Data and Results](#data-and-results)
- [Questions](#questions)

## Installation

To run this project, ensure you have Python 3.7 or higher installed. Use the following command to install the necessary packages:

```bash
pip install -r requirements.txt
```

## Data and Results
All the data used in this project are stored in the `data` folder.
The notebook with more information and results is inside the notebooks folder in the `EDA.ipynb` file.

## Questions

### 1. Steps to Solve the Problem

1. **Define the Problem and Objectives:**  Clearly understand the business goal: to minimize maintenance costs through accurate truck defect prediction.
2. **Data Collection and Preparation:** Gather historical data (denoted as `df_previous`) and current data (`df_present`), ensuring data quality and consistency.
3. **Exploratory Data Analysis (EDA):** Perform EDA to comprehend data distribution, identify outliers, and pinpoint relevant features for prediction.
4. **Data Splitting:** Divide `df_previous` into training and validation sets for model development and hyperparameter tuning.
5. **Model Selection and Training:** Consider RandomForest as an initial model, train it using the training set.
6. **Hyperparameter Tuning:** Employ GridSearchCV to fine-tune the model's hyperparameters, enhancing predictive performance.
7. **Model Evaluation:** Assess the model using metrics like accuracy, precision, recall and F1-scoreto gauge its effectiveness.
8. **Cost Estimation:** Calculate total expected cost based on business rules and predicted probabilities (consider maintenance vs. potential defect costs).

### 2. Technical Data Science Metrics

* **F1-score:** Balances precision (correct positive rate) and recall (true positive rate) to minimize false positives (unnecessary maintenance) and false negatives (undetected defects).

### 3. Business Metric

* **Total Expected Cost:** Captures the costs of unnecessary maintenance, preventive maintenance for defective trucks, and non-maintenance for undetected defects.

### 4. Relationship Between Technical and Business Metrics

Strong technical metrics (high F1-score, and specially recall) lead to accurate predictions, directly impacting business outcomes by minimizing unnecessary maintenance and preventing costly undetected defects.

### 5. Customer Database Analyses

* **Descriptive Analysis:** To understand the basic characteristics of the data.
* **Predictive Analysis:** To predict future outcomes based on historical data.
* **Segmentation Analysis:** To identify different segments of trucks based on their maintenance needs.
* **Anomaly Detection:** To identify outliers or unusual patterns in the data.

### 6. Dimensionality Reduction Techniques

* **Feature Selection Techniques:** Such as Recursive Feature Elimination (RFE) to select the most important features.

### 7. Variable Selection Techniques for Predictive Models

* **Correlation Analysis:** To identify and remove highly correlated features.
* **Feature Importance from Models:** Using models like Random Forest or XGBoost to identify important features.
  
### 8. Predictive Models

* **XGBoost:** Known for its high performance in classification tasks.
* **Random Forest:** For its robustness and ability to handle a large number of features.
* **Logistic Regression:** As a baseline model for comparison.
  
### 9. Assessing Trained Models

Compare models using cross-validated scores on metrics like F1-score and specially recall. The model with the best combination of these measures and lowest total expected cost would be considered the best.

### 10. Model Result Explanation

* **By track the cost:** Calculate the cost and compare for each model and without the predection to compare the difference beetwen each one of them. and by feature importance is also possible to find which variable are the most important.

### 11. Assessing Financial Impact of the Model

* **Estimate Cost Savings:** Quantify cost reductions from fewer unnecessary maintenance instances and avoided undetected defects.
* **Calculate ROI (Return on Investment):** Demonstrate the model's financial benefits to stakeholders by calculating the ROI based on cost savings and implementation costs.

### 12. Hyperparameter Optimization Techniques

* **Grid Search:** To systematically search through a predefined set of hyperparameters, evaluating each combination to find the optimal settings.

### 13. Risks and Precautions Before Production

* **Data Privacy and Security:** Ensure compliance with relevant regulations and secure handling of data throughout the entire process.
* **Model Fairness and Bias:**  Check for and mitigate any biases in the model that could lead to unfair predictions for certain truck segments.
* **Overfitting and Generalization:** Ensure the model performs well on unseen data by employing techniques like regularization and cross-validation to prevent overfitting to the training data.
* **Model Monitoring:** Have a plan for continuous monitoring of the model's performance to detect any degradation in accuracy over time.

### 14. Model Deployment

* **Deployment Strategy:** Develop a deployment strategy that aligns with client infrastructure. Considering cloud services like AWS, Azure, or GCP for scalability and ease of management.
* **Integration:** Integrate the model into the existing systems for real-time or batch predictions. This may involve building APIs or data pipelines.
* **Scalability and Reliability:** Ensure the deployment can handle the expected volume of data and predictions without compromising performance or availability.

### 15. Model Monitoring in Production

* **Performance Metrics:** Continuously track performance metrics like accuracy, precision, recall, and F1-score to identify any significant drops that might necessitate retraining.
* **Data Drift:** Monitor for changes in the data distribution over time (data drift). Significant data drift can degrade model performance, so it's important to have mechanisms to detect and address it.
* **Prediction Accuracy:** Regularly compare model predictions with actual outcomes to assess the model's ongoing effectiveness.

### 16. Retraining the Model

* **Performance Thresholds:** Set up performance thresholds that trigger retraining if metrics fall below a certain level.
* **Periodic Reviews:** Schedule periodic reviews of model performance to proactively assess the need for retraining.
* **Data Drift Detection:**  As mentioned earlier, data drift can necessitate retraining. Implement mechanisms to detect significant data drift and trigger retraining processes accordingly.



