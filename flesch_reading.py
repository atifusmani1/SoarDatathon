import pandas as pd
import textstat as ts

def flesch(text):
    return ts.flesch_reading_ease(text)


def add_textstat(df):
    df = df.copy()

    df['flesch_a'] = df["response_a"].apply(flesch)
    df['flesch_b'] = df["response_b"].apply(flesch)

    df["flesch_ratio"] = df['flesch_a'] / df['flesch_b']

    return df    