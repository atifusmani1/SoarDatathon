import pandas as pd

def add_length_ratio(df):
    df['response_length_ratio'] = df.apply(lambda row: len(row['response_a']) / len(row['response_b']), axis=1)
    return df
