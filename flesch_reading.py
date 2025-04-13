import pandas as pd
import textstat as ts
from tqdm import tqdm
tqdm.pandas()

def flesch(text):
    return ts.flesch_reading_ease(text)


def add_flesch_reading(df):
    df = df.copy()

    df['flesch_a'] = df["response_a"].apply(flesch)
    df['flesch_b'] = df["response_b"].apply(flesch)

    df["flesch_ratio"] = df['flesch_a'] / df['flesch_b']

    return df    

