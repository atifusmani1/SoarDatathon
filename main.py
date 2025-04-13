import pandas as pd
from build_feature_df import build_central_df
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train_path = '/Users/deshawnwalker/Desktop/datathon/train.csv'
test_path = '/Users/deshawnwalker/Desktop/datathon/test.csv'

training_df = pd.read_csv(train_path)
training_df = build_central_df(training_df)

print(training_df.columns) 