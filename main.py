import pandas as pd
from length_metric import add_length_ratio
from flesch_reading import add_flesch_reading
from sentiment_analyzer import add_sentiment

csv_path = '/Users/deshawnwalker/Desktop/datathon/train.csv'

llm_training_df = pd.read_csv(csv_path)

llm_training_df = add_length_ratio(llm_training_df)
llm_training_df = add_flesch_reading(llm_training_df)
llm_training_df = add_sentiment(llm_training_df)
