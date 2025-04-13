import pandas as pd
from length_metric import add_length_ratio
from flesch_reading import add_flesch_reading
from sentiment_analyzer import add_sentiment
from lexical_diversity import add_lexical_diversity_score
from update_response_variable import update_response

def build_central_df(df):
    df = add_length_ratio(df)
    df = add_flesch_reading(df)
    df = add_sentiment(df)
    df = add_lexical_diversity_score(df)
    df = update_response(df)
    return df

