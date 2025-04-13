import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return analyzer.polarity_scores(str(text))['compound']

def add_sentiment(df):
    df['response_a_sentiment'] = df['response_a'].apply(get_sentiment)
    df['response_b_sentiment'] = df['response_b'].apply(get_sentiment)
    df['prompt_sentiment'] = df['prompt'].apply(get_sentiment)
    return df
