import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Remove unnecessary columns
df = df.drop(columns=[
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
])

X = df.drop(columns=["Attrition"])
y = df["Attrition"].map({"No": 0, "Yes": 1})

# Identify feature types
categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features),
])

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    ),
}

print("\n===== MODEL COMPARISON =====")

for name, classifier in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Attrition F1-score:", round(
        f1_score(y_test, predictions, zero_division=0), 4
    ))
    print("Precision:", round(
        precision_score(y_test, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, zero_division=0), 4
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
print("\nModel comparison completed!")
