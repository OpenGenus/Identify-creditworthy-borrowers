# Identify creditworthy borrowers
![image](https://github.com/deshpanda/Identify-creditworthy-borrowers/assets/96617520/0aac8c3b-90b3-4cd8-9f70-691447e4e053)


## Overview
This Jupyter Notebook presents a comprehensive approach to predict whether a credit applicant is "good" or "bad" based on the provided attributes. The "good" or "bad" classification represents the creditworthiness of the applicant, which is crucial for financial institutions to make informed lending decisions and mitigate potential risks.

## Table of Contents
* Introduction
* Dataset
* Feature Engineering
* Machine Learning Models
* Evaluation
* Usage
* Dependencies

## Introduction
Credit risk assessment is a critical task in the financial sector, with significant implications for lending institutions and borrowers. This notebook provides an in-depth analysis of a credit risk dataset, exploring various features that may influence creditworthiness. By leveraging feature engineering, the aim is to enhance the dataset's predictive power and improve model performance.

## Dataset
The Statlog German Credit Dataset is a widely used and publicly available dataset that provides valuable insights into credit risk assessment and lending practices. The dataset was created by Prof. Dr. Hans Hofmann in 1994 and is available through the UCI Machine Learning Repository, a renowned repository for machine learning datasets.

The dataset comprises information on credit applications from German banks and consists of 20 attributes, including both numerical and categorical features, making it suitable for various machine learning algorithms. Among the key attributes are the status of existing checking accounts, duration of credit, credit history, purpose of the credit, credit amount, savings account or bonds, present employment status, personal status, and age, among others.

## Feature Engineering
Feature engineering is a critical step to preprocess the dataset and create informative features that contribute to the predictive model's accuracy. In this notebook, we delve into data cleaning, handling missing values and encoding categorical variables. The feature engineering process aims to optimize the dataset for better model performance.

## Machine Learning Models
This notebook explores various machine learning models to predict credit risk based on the transformed dataset. A range of algorithms, such as Logistic Regression, Decision Tree, K-Nearest Neighbors(KNN), Support Vector Machine(SVM), and Multilayer Perceptron(MLP), are implemented and compared for their predictive capabilities. The performance of each model is thoroughly evaluated using appropriate metrics.

## Evaluation
The effectiveness of the models is determined through a detailed evaluation process, considering metrics like accuracy, precision, recall, and F1-score. Furthermore, the notebook includes understanding the models' behavior and performance.

## Usage
To reproduce the results and analysis presented in this notebook, you will need to have Jupyter Notebook installed on your local machine or access to a platform that supports Jupyter Notebook files (e.g., Kaggle Notebooks, Google Colab).

* Clone the repository to your local machine.
* Install the required dependencies listed in the next section.
* Open the Jupyter Notebook and run the cells sequentially.

Please note that you might need to adjust file paths or modify the code slightly if you are running the notebook in a different environment or using a different dataset.

## Dependencies
The notebook relies on the following Python libraries for data manipulation, visualization, and machine learning:
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

You can install the required libraries by running the following command in your terminal:

```
pip install numpy pandas matplotlib seaborn scikit-learn
```
