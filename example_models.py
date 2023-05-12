from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {"logistic_regression": LogisticRegression(verbose=True, max_iter=1000, random_state=42), 
          "random_forest": RandomForestClassifier(verbose=True, n_estimators=120, criterion="gini"),
          "xgb": XGBClassifier(random_state=42)}