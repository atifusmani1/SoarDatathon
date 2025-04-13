import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train_path = '/Users/deshawnwalker/Desktop/datathon/train.csv'
test_path = '/Users/deshawnwalker/Desktop/datathon/test.csv'

training_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

training_df = add_length_ratio(training_df)
training_df = add_flesch_reading(training_df)
training_df = add_sentiment(training_df)
training_df = add_lexical_diversity_score(training_df)

test_df = add_length_ratio(test_df)
test_df = add_flesch_reading(test_df)
test_df = add_sentiment(test_df)
test_df = add_lexical_diversity_score(test_df)



