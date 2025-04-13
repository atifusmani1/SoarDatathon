import pandas as pd

# Load the CSV file
full_train_df = pd.read_csv('C:\\Users\\16164\\Documents\\SoarDatathon\\lmsys-chatbot-arena\\train.csv')

non_tie_df = full_train_df[(full_train_df['winner_tie'] == 0) & 
                           ((full_train_df['winner_model_a'] == 1) | (full_train_df['winner_model_b'] == 1))].copy()

# print(non_tie_df.head(5))

def text_length(text):
    return len(text)

# For Model A wins: winning response from response_a and losing from response_b
non_tie_df.loc[non_tie_df['winner_model_a'] == 1, 'winning_length'] = non_tie_df.loc[non_tie_df['winner_model_a'] == 1, 'response_a'].apply(text_length)
non_tie_df.loc[non_tie_df['winner_model_a'] == 1, 'losing_length'] = non_tie_df.loc[non_tie_df['winner_model_a'] == 1, 'response_b'].apply(text_length)

# For Model B wins: winning response from response_b and losing from response_a
non_tie_df.loc[non_tie_df['winner_model_b'] == 1, 'winning_length'] = non_tie_df.loc[non_tie_df['winner_model_b'] == 1, 'response_b'].apply(text_length)
non_tie_df.loc[non_tie_df['winner_model_b'] == 1, 'losing_length'] = non_tie_df.loc[non_tie_df['winner_model_b'] == 1, 'response_a'].apply(text_length)

# Display the first few rows with the new columns
# print(non_tie_df[['response_a', 'response_b', 'winning_length', 'losing_length']].head(5))

# Losing responses statistics
sum_losing_lens = non_tie_df['losing_length'].sum()
losing_response_length_avg = non_tie_df['losing_length'].mean()


# Winning responses statistics
sum_winning_lens = non_tie_df['winning_length'].sum()
winning_response_length_avg = non_tie_df['winning_length'].mean()

# printed results
print("Average losing response length:", losing_response_length_avg)

print("Average winning response length:", winning_response_length_avg)

#edit main df
full_train_df.loc[non_tie_df.index, "winning_length"] = non_tie_df["winning_length"]

full_train_df.loc[non_tie_df.index, "losing_length"] = non_tie_df["losing_length"]

print(full_train_df.head(5))