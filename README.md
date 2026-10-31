# AttritionAI | Employee Analytics

**A Machine Learning-Based Employee Attrition Analysis and Prediction System**

## 📌 Project Overview

AttritionAI is a web-based Employee Analytics application developed using Python and Machine Learning. It analyzes employee data to identify attrition patterns, understand workforce trends, and predict potential employee turnover risk.

The application provides an interactive dashboard, detailed employee analytics, and an employee attrition risk prediction system to help organizations make data-driven decisions.

## ✨ Key Features

- **Interactive Dashboard:** Displays overall employee statistics and workforce insights.
- **Employee Analysis:** Provides detailed analysis of employee attrition patterns.
- **Data Visualization:** Includes 9 interactive charts for understanding workforce trends.
- **Department-wise Analysis:** Analyzes employee distribution and attrition across departments.
- **Demographic Analysis:** Explores attrition based on age, gender, income, and other factors.
- **Automatic Key Findings:** Generates insights from employee data.
- **Model Performance Evaluation:** Displays Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- **Confusion Matrix:** Shows the classification results of the prediction model.
- **Risk Prediction:** Predicts potential employee attrition risk based on employee details.
- **About Project:** Provides information about the application.

## 🛠️ Technologies Used

### 1. Programming Language
- **Python** – Main programming language used for data processing, machine learning, and backend development.

### 2. Backend Development
- **Flask** – Used to build the web application and handle routing and prediction requests.

### 3. Frontend Development
- **HTML5** – Structures the web pages.
- **CSS3** – Provides styling, layouts, and user interface design.
- **JavaScript** – Handles interactive dashboard functionality.
- **Chart.js** – Creates interactive charts and data visualizations.

### 4. Data Analysis and Processing
- **Pandas** – Used for loading, cleaning, and analyzing employee datasets.
- **NumPy** – Supports numerical and data operations.

### 5. Machine Learning
- **Scikit-learn** – Used for data preprocessing, model evaluation, and prediction.
- **Logistic Regression** – Machine learning algorithm used for employee attrition prediction.
- **Joblib** – Used to save and load the trained machine learning model.

### 6. Development Tools
- **Visual Studio Code** – Used for writing and managing project code.
- **Git and GitHub** – Used for version control and project hosting.

## 🤖 Machine Learning Model

The project uses a trained **Logistic Regression** model to predict employee attrition.

The model takes employee-related information as input and predicts whether an employee has a potential attrition risk.

### Model Evaluation Metrics

The model is evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

A Confusion Matrix is also used to understand the model's classification performance.

## 📊 Employee Analytics

The Employee Analysis page contains nine visualizations:

1. Employee Attrition Distribution
2. Employees by Department
3. Attrition by Department
4. Gender Distribution
5. Job Satisfaction vs Attrition
6. Attrition by Age Group
7. Attrition by Monthly Income
8. Overtime vs Attrition
9. Attrition by Years at Company

These visualizations help identify patterns and relationships within employee data.

## 📁 Project Structure

```text
ChurnAI/
│
├── app.py
├── train.py
├── predict.py
├── compare_models.py
├── eda.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   └── attrition_model.pkl
│
└── templates/
    ├── about.html
    ├── analysis.html
    ├── dashboard.html
    ├── index.html
    └── result.html
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/anshika55707-lgtm/AttritionAI.git
```

### 2. Navigate to the Project Folder

```bash
cd ChurnAI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Flask Application

```bash
python app.py
```

### 7. Open in Browser

Visit:

```text
http://127.0.0.1:5000/
```

## 🎯 Project Objective

The main objective of AttritionAI is to analyze employee attrition patterns and apply machine learning techniques to identify potential employee turnover risks.

It aims to demonstrate how data analytics and machine learning can support better workforce analysis and informed organizational decision-making.



**Developed as an AI & Data Science project.**
## 📸 Project Screenshots

### 1. Dashboard
![Dashboard](screenshots/Screenshot%20%28206%29.png)

### 2. Employee Analysis
![Employee Analysis](screenshots/Screenshot%20%28207%29.png)
## 📊 Model Performance

The Employee Attrition Prediction system uses Logistic Regression to predict employee attrition.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 75.17% |
| Precision | 34.88% |
| Recall | 63.83% |
| F1-Score | 45.11% |
| ROC-AUC Score | 80.32% |

The model evaluation helps assess its ability to identify employees who may be at risk of leaving the organization.

### 3. Risk Prediction
![Risk Prediction](screenshots/Screenshot%20%28208%29.png)

### 4. Prediction Result
![Prediction Result](screenshots/Screenshot%20%28209%29.png)
