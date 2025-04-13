import pandas as pd

def update_response(df):
    df = df.copy()

    df['winner'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)

    df['winner'] = df['winner'].map({
        'winner_model_a': 'a',
        'winner_model_b': 'b',
        'winner_tie': 'tie'
    })

    return df
