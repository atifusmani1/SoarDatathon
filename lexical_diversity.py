### this file calculates the lexical diversity metrics for each response, using the text-token ration

## load data

import pandas as pd

train_data = pd.read_csv(r"C:\Users\zayaa\Downloads\SoarDatathon\data\train.csv")
print(train_data.head())

### prepare text 

from lexical_diversity import lex_div as ld

print(train_data.columns) # print column names

text_a = train_data['response_a']
text_b = train_data['response_b']
print(text_a[10:])

type(text_a) # returns that the data type is series; needs to be string

## tokenize (split sentence into words)

import nltk
from nltk.tokenize import word_tokenize

# Step 1: clean the responses by removing outer brackets and quotes
def strip_brackets(x):
    x = str(x)
    return x[2:-2] if x.startswith('["') and x.endswith('"]') else x

train_data['tok_a'] = text_a.apply(strip_brackets)
train_data['tok_b'] = text_b.apply(strip_brackets)

# Step 2: Tokenize the cleaned strings
train_data['tok_a'] = train_data['tok_a'].astype(str).apply(word_tokenize)
train_data['tok_b'] = train_data['tok_b'].astype(str).apply(word_tokenize)

# View test result for the tokenized b responses
print(train_data[['response_b', 'tok_b']].head())

# lemmatize text to treat different forms of word as one (ex. running and run)

train_data['lem_a'] = train_data['tok_a'].astype(str).apply(ld.flemmatize)
train_data['lem_b'] = train_data['tok_b'].astype(str).apply(ld.flemmatize)
print(train_data['lem_b'].head())

## calculate the lexical diversity using TTR metric

train_data['log_ttr_a'] = train_data['lem_a'].apply(ld.log_ttr)
train_data['log_ttr_b'] = train_data['lem_b'].apply(ld.log_ttr)
print(train_data[['log_ttr_a','log_ttr_b']].head())

# calculate the ratio (a/b)  

train_data['ttr_ratio'] = train_data['log_ttr_a']/train_data['log_ttr_b']
train_data['ttr_ratio']

### function to get lexity diversity score ratio

def add_lexical_diversity_score(df):
    import pandas as pd

    # define text responses from a and b
    text_a = df['response_a']
    text_b = df['response_b']

    ## tokenize (split sentence into words)
    import nltk
    from nltk.tokenize import word_tokenize
    
    # Step 1: clean the responses by removing outer brackets and quotes if present
    def strip_brackets(x):
        x = str(x)
        return x[2:-2] if x.startswith('["') and x.endswith('"]') else x
    
    df['tok_a'] = text_a.apply(strip_brackets)
    df['tok_b'] = text_b.apply(strip_brackets)
    
    # Step 2: Tokenize the cleaned strings
    df['tok_a'] = df['tok_a'].astype(str).apply(word_tokenize)
    df['tok_b'] = df['tok_b'].astype(str).apply(word_tokenize)
        
    ## lemmatize text to deal with capitalization and 
    ## treat different forms of word as one (ex. running and run)
    
    from lexical_diversity import lex_div as ld

    df['lem_a'] = df['tok_a'].astype(str).apply(ld.flemmatize)
    df['lem_b'] = df['tok_b'].astype(str).apply(ld.flemmatize)
    
    ## calculate the lexical diversity using TTR metric
    df['log_ttr_a'] = df['lem_a'].apply(ld.log_ttr)
    df['log_ttr_b'] = df['lem_b'].apply(ld.log_ttr)
    
    # calculate the ratio (a/b)  
    
    df['ttr_ratio'] = df['log_ttr_a']/df['log_ttr_b']
    return df

print(add_lexical_diversity_score(train_data))



