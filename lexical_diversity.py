### this file calculates the lexical diversity metrics for each response, using the text-token ration

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


