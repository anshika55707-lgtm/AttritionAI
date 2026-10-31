
import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET INFORMATION")
df.info()

print("\nMISSING VALUES")
print(df.isnull().sum().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nATTRITION DISTRIBUTION")
print(df["Attrition"].value_counts(normalize=True) * 100)

print("\nUNIQUE VALUES IN EACH COLUMN")
print(df.nunique())

# Check constant columns
constant_columns = [
    col for col in df.columns
    if df[col].nunique() <= 1
]

print("\nCONSTANT COLUMNS")
print(constant_columns)

# Check unique employee IDs
print("\nEMPLOYEE NUMBER UNIQUE COUNT")
print(df["EmployeeNumber"].nunique())
