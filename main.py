import pandas as pd
from length_metric import add_length_ratio
from flesch_reading import add_flesch_reading
from sentiment_analyzer import add_sentiment

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

train_path = '/Users/deshawnwalker/Desktop/datathon/train.csv'
test_path = '/Users/deshawnwalker/Desktop/datathon/test.csv'

training_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

training_df = add_length_ratio(training_df)
training_df = add_flesch_reading(training_df)
training_df = add_sentiment(training_df)

test_df = add_length_ratio(test_df)
test_df = add_flesch_reading(test_df)
test_df = add_sentiment(test_df)


