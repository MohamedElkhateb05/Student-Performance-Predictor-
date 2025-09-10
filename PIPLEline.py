import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


X = df2.drop("Exam_Score", axis=1)
y = df2["Exam_Score"]


num_features = ["Hours_Studied", "Attendance", "Sleep_Hours", 
                "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]

cat_features = ["Parental_Involvement", "Access_to_Resources", 
                "Extracurricular_Activities", "Motivation_Level",
                "Internet_Access", "Family_Income", "Teacher_Quality", 
                "School_Type", "Peer_Influence", "Learning_Disabilities",
                "Parental_Education_Level", "Distance_from_Home", "Gender"]


num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])


cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

print("Train R^2:", model.score(X_train, y_train))
print("Test R^2:", model.score(X_test, y_test))
