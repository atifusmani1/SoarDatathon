import pandas as pd
import textstat as ts

def flesch(text):
    return ts.flesch_reading_ease(text)

def add_textstat(df):
    df = df.copy()
    df["flesch_prompt"] = df["prompt"].apply(flesch)
    df["flesch_response_a"] = df["response_a"].apply(flesch)
    df["flesch_response_b"] = df["response_b"].apply(flesch)
    
    return df
