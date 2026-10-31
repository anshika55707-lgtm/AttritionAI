
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/attrition_model.pkl")

# Create sample employee data
employee = {
    "Age": 29,
    "BusinessTravel": "Travel_Frequently",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 2,
    "Gender": "Male",
    "HourlyRate": 60,
    "JobInvolvement": 2,
    "JobLevel": 1,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 2,
    "MaritalStatus": "Single",
    "MonthlyIncome": 3000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 3,
    "OverTime": "Yes",
    "PercentSalaryHike": 13,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 2,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 5,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 2,
    "YearsInCurrentRole": 1,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 1,
}

# Convert employee data into a DataFrame
employee_df = pd.DataFrame([employee])

# Predict attrition
prediction = model.predict(employee_df)[0]
probability = model.predict_proba(employee_df)[0][1]

# Display results
print("\n===== EMPLOYEE ATTRITION PREDICTION =====")

if prediction == 1:
    print("Prediction: Employee may leave")
else:
    print("Prediction: Employee may stay")

print(f"Predicted attrition probability: {probability:.2%}")
print("\nPrediction completed successfully!")
