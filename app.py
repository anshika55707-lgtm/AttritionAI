

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
app = Flask(__name__)

# Load the existing trained model
model = joblib.load("models/attrition_model.pkl")

# Load employee dataset
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(DATA_PATH)

# Dashboard statistics from actual dataset
total_employees = len(df)
attrition_count = int((df["Attrition"] == "Yes").sum())
stayed_count = int((df["Attrition"] == "No").sum())
attrition_rate = round((attrition_count / total_employees) * 100, 2)

# Model evaluation using the same split as train.py
drop_columns = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

model_df = df.drop(columns=drop_columns)
X = model_df.drop(columns=["Attrition"])
y = model_df["Attrition"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Evaluate the saved model without retraining it
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

confusion_matrix_data = {
    "true_negative": int(cm[0][0]),
    "false_positive": int(cm[0][1]),
    "false_negative": int(cm[1][0]),
    "true_positive": int(cm[1][1])
}

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
    "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
    "precision": round(
        precision_score(y_test, y_pred, zero_division=0) * 100, 2
    ),
    "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
    "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2)
}

# Department statistics
department_counts = (
    df["Department"].value_counts().to_dict()
)

department_attrition = (
    df[df["Attrition"] == "Yes"]["Department"]
    .value_counts()
    .to_dict()
)

# Shared data for dashboard and analysis pages
dashboard_data = {
    "total_employees": total_employees,
    "attrition_count": attrition_count,
    "stayed_count": stayed_count,
    "attrition_rate": attrition_rate,
    "department_counts": department_counts,
    "department_attrition": department_attrition,
    "metrics": metrics
}


# 1. Dashboard
@app.route("/")
@app.route("/dashboard")
def home():
    return render_template("dashboard.html", **dashboard_data)


# 2. Employee Analysis

@app.route("/analysis")
def analysis():
    import os

    csv_path = os.path.join(
        app.root_path,
        "data",
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    df = pd.read_csv(csv_path)

    total_employees = int(len(df))

    # Normalize the target values for counting.
    attrition_values = df["Attrition"].astype(str).str.strip().str.lower()
    left_count = int(attrition_values.isin(["yes", "1", "true"]).sum())
    stayed_count = total_employees - left_count

    attrition_rate = (
        round((left_count / total_employees) * 100, 2)
        if total_employees else 0
    )

    department_counts = (
        df["Department"].fillna("Unknown")
        .value_counts()
        .to_dict()
    )

    department_attrition = {}
    for department in df["Department"].fillna("Unknown").unique():
        department_df = df[
            df["Department"].fillna("Unknown") == department
        ]
        department_attrition[str(department)] = {
            "left": int(
                department_df["Attrition"]
                .astype(str).str.strip().str.lower()
                .isin(["yes", "1", "true"]).sum()
            ),
            "stayed": int(
                len(department_df) -
                department_df["Attrition"]
                .astype(str).str.strip().str.lower()
                .isin(["yes", "1", "true"]).sum()
            )
        }

    gender_counts = (
        df["Gender"].fillna("Unknown")
        .value_counts()
        .to_dict()
    ) if "Gender" in df.columns else {}

    satisfaction_attrition = {}
    if "JobSatisfaction" in df.columns:
        for level in sorted(df["JobSatisfaction"].dropna().unique()):
            subset = df[df["JobSatisfaction"] == level]
            values = subset["Attrition"].astype(str).str.strip().str.lower()

            satisfaction_attrition[str(level)] = {
                "left": int(values.isin(["yes", "1", "true"]).sum()),
                "stayed": int(len(subset) - values.isin(["yes", "1", "true"]).sum())
            }
                # 1. Attrition by Age Group
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65"]
    )

    age_attrition = {}
    for group in df["AgeGroup"].cat.categories:
        subset = df[df["AgeGroup"] == group]
        values = subset["Attrition"].astype(str).str.strip().str.lower()

        age_attrition[str(group)] = {
            "left": int(values.isin(["yes", "1", "true"]).sum()),
            "stayed": int(len(subset) - values.isin(["yes", "1", "true"]).sum())
        }

    # 2. Attrition by Monthly Income
    df["IncomeGroup"] = pd.cut(
        df["MonthlyIncome"],
        bins=[0, 3000, 6000, 10000, float("inf")],
        labels=["0-3K", "3K-6K", "6K-10K", "10K+"]
    )

    income_attrition = {}
    for group in df["IncomeGroup"].cat.categories:
        subset = df[df["IncomeGroup"] == group]
        values = subset["Attrition"].astype(str).str.strip().str.lower()

        income_attrition[str(group)] = {
            "left": int(values.isin(["yes", "1", "true"]).sum()),
            "stayed": int(len(subset) - values.isin(["yes", "1", "true"]).sum())
        }

    # 3. Overtime vs Attrition
    overtime_attrition = {}
    for group in df["OverTime"].dropna().unique():
        subset = df[df["OverTime"] == group]
        values = subset["Attrition"].astype(str).str.strip().str.lower()

        overtime_attrition[str(group)] = {
            "left": int(values.isin(["yes", "1", "true"]).sum()),
            "stayed": int(len(subset) - values.isin(["yes", "1", "true"]).sum())
        }

    # 4. Attrition by Years at Company
    df["TenureGroup"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 1, 5, 10, 20, float("inf")],
        labels=["0-1", "2-5", "6-10", "11-20", "20+"]
    )

    tenure_attrition = {}
    for group in df["TenureGroup"].cat.categories:
        subset = df[df["TenureGroup"] == group]
        values = subset["Attrition"].astype(str).str.strip().str.lower()

        tenure_attrition[str(group)] = {
            "left": int(values.isin(["yes", "1", "true"]).sum()),
            "stayed": int(len(subset) - values.isin(["yes", "1", "true"]).sum())
        }
        
    # Calculate data-driven analytics insights
    def summarize_attrition_groups(group_data):
        valid_groups = {
            str(group): counts
            for group, counts in group_data.items()
            if counts["left"] + counts["stayed"] > 0
        }

        if not valid_groups:
            return {
                "highest_rate_group": "N/A",
                "highest_rate": 0,
                "most_left_group": "N/A",
                "most_left": 0
            }

        rates = {
            group: round(
                counts["left"] /
                (counts["left"] + counts["stayed"]) * 100,
                2
            )
            for group, counts in valid_groups.items()
        }

        highest_rate_group = max(rates, key=rates.get)
        most_left_group = max(
            valid_groups,
            key=lambda group: valid_groups[group]["left"]
        )

        return {
            "highest_rate_group": highest_rate_group,
            "highest_rate": rates[highest_rate_group],
            "most_left_group": most_left_group,
            "most_left": valid_groups[most_left_group]["left"]
        }

    analytics_insights = {
        "age": summarize_attrition_groups(age_attrition),
        "income": summarize_attrition_groups(income_attrition),
        "overtime": summarize_attrition_groups(overtime_attrition),
        "tenure": summarize_attrition_groups(tenure_attrition),
       
    }
        # Generate automatic key findings
    key_findings = [
        f"Age group {analytics_insights['age']['highest_rate_group']} "
        f"has the highest attrition rate by age "
        f"({analytics_insights['age']['highest_rate']}%).",

        f"Monthly income group "
        f"{analytics_insights['income']['highest_rate_group']} "
        f"has the highest attrition rate by income "
        f"({analytics_insights['income']['highest_rate']}%).",

        f"Overtime group "
        f"{analytics_insights['overtime']['highest_rate_group']} "
        f"has the highest attrition rate "
        f"({analytics_insights['overtime']['highest_rate']}%).",

        f"Employees in the "
        f"{analytics_insights['tenure']['highest_rate_group']} years-at-company "
        f"group have the highest attrition rate by tenure "
        f"({analytics_insights['tenure']['highest_rate']}%)."
    ]

    return render_template (
        "analysis.html",
        total_employees=total_employees,
        attrition_count=left_count,
        stayed_count=stayed_count,
        attrition_rate=attrition_rate,
        department_counts=department_counts,
        department_attrition=department_attrition,
        gender_counts=gender_counts,
        satisfaction_attrition=satisfaction_attrition,
        age_attrition=age_attrition,
        income_attrition=income_attrition,
        overtime_attrition=overtime_attrition,
        tenure_attrition=tenure_attrition,
        analytics_insights=analytics_insights,
        confusion_matrix_data=confusion_matrix_data,
                metrics=metrics,
        key_findings=key_findings
    )
    

    return render_template("analysis.html", **dashboard_data)


# 3. Risk Prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("index.html")

    try:
        employee = {
            "Age": int(request.form["age"]),
            "BusinessTravel": request.form["travel"],
            "DailyRate": 800,
            "Department": request.form["department"],
            "DistanceFromHome": 10,
            "Education": 3,
            "EducationField": "Life Sciences",
            "EnvironmentSatisfaction": 2,
            "Gender": "Male",
            "HourlyRate": 60,
            "JobInvolvement": 2,
            "JobLevel": 1,
            "JobRole": request.form["jobRole"],
            "JobSatisfaction": int(request.form["satisfaction"]),
            "MaritalStatus": request.form["marital"],
            "MonthlyIncome": int(request.form["income"]),
            "MonthlyRate": 15000,
            "NumCompaniesWorked": 3,
            "OverTime": request.form["overtime"],
            "PercentSalaryHike": 13,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 5,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 2,
            "YearsAtCompany": int(request.form["years"]),
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 1
        }

        employee_df = pd.DataFrame([employee])

        prediction = int(model.predict(employee_df)[0])
        probability = float(model.predict_proba(employee_df)[0][1])

        result = (
            "Potential Attrition Risk"
            if prediction == 1
            else "Lower Predicted Attrition Risk"
        )

        return render_template(
            "result.html",
            result=result,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        print("PREDICTION ERROR:", repr(e))
        return "Prediction failed. Check the VS Code terminal.", 400


# 4. About Project
@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
