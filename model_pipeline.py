# 📦 Imports
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 📁 Load data
train = pd.read_csv("Train_Data.csv")
test = pd.read_csv("Test_Data.csv")
sample_submission = pd.read_csv("Sample_Submission.csv")

# 🧹 Basic cleaning
train.drop(columns=["SEQN"], inplace=True)
test_ids = test["SEQN"]
test.drop(columns=["SEQN"], inplace=True)

train["target"] = train["age_group"].map({"Adult": 0, "Senior": 1})
train.dropna(subset=["target"], inplace=True)

y = train["target"]
X = train.drop(columns=["age_group", "target"])
X_test = test.copy()

X.columns = X.columns.str.lower()
X_test.columns = X_test.columns.str.lower()

# 🔧 Feature Engineering
X["glucose_insulin_ratio"] = X["lbxglu"] / (X["lbxin"] + 1)
X_test["glucose_insulin_ratio"] = X_test["lbxglu"] / (X_test["lbxin"] + 1)

X["bmi_log"] = np.log1p(X["bmxbmi"])
X_test["bmi_log"] = np.log1p(X_test["bmxbmi"])

X["glucose_high"] = (X["lbxglu"] > 125).astype(int)
X_test["glucose_high"] = (X_test["lbxglu"] > 125).astype(int)

X["bmi_high"] = (X["bmxbmi"] > 30).astype(int)
X_test["bmi_high"] = (X_test["bmxbmi"] > 30).astype(int)

# 🧽 Drop low-variance columns
low_var_cols = X.columns[X.nunique() <= 1]
X.drop(columns=low_var_cols, inplace=True)
X_test.drop(columns=low_var_cols, inplace=True)

# 🧱 Preprocessing Pipelines
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ⚙️ Preprocess Data
X_pre = preprocessor.fit_transform(X)
X_test_pre = preprocessor.transform(X_test)

X_pre = pd.DataFrame(X_pre)
X_test_pre = pd.DataFrame(X_test_pre)

# 🤖 Model Definitions
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=1.5)
lgbm = LGBMClassifier(random_state=42, class_weight='balanced')
cb = CatBoostClassifier(verbose=0, random_state=42)

stacked_model = StackingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('cb', cb)
    ],
    final_estimator=LGBMClassifier(random_state=42),
    cv=5,
    n_jobs=-1
)

# 🔁 Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X_pre.reset_index(drop=True, inplace=True)
y = y.reset_index(drop=True)

cv_scores = []

for train_idx, val_idx in cv.split(X_pre, y):
    X_train, X_val = X_pre.iloc[train_idx], X_pre.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    stacked_model.fit(X_train, y_train)
    val_preds = stacked_model.predict(X_val)
    score = f1_score(y_val, val_preds)
    cv_scores.append(score)

print("✅ Average CV F1 Score:", round(np.mean(cv_scores), 4))

# 🏁 Train on full data
stacked_model.fit(X_pre, y)
final_preds = stacked_model.predict(X_test_pre)

# 📤 Prepare Submission
submission = sample_submission.copy()
submission["age_group"] = final_preds
submission.to_csv("submission.csv", index=False)

print("📁 submission.csv is ready for upload.")
