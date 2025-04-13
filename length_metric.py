import pandas as pd

full_train_df = pd.read_csv('C:\\Users\\16164\\Documents\\SoarDatathon\\lmsys-chatbot-arena\\train.csv')

def add_length_ratio(df):
    df['a_b_ratio'] = df.apply(lambda row: len(row['response_a']) / len(row['response_b']), axis=1)
    return df

full_train_df = add_length_ratio(full_train_df)

print(full_train_df.head(5))
