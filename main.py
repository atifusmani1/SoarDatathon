import pandas as pd
import numpy as np
from build_feature_df import build_central_df
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, log_loss, mean_squared_error
import matplotlib.pyplot as plt 


# training_df = build_central_df(training_df) to get the updated training df

training_df = pd.read_csv(r"c:\Users\zayaa\Downloads\SoarDatathon\data\training_data_updated.csv")

print(training_df.columns)

# remove rows with missing data
training_df = training_df.dropna()
training_df.isnull().sum()

# remove values that have infinity
mask = np.isinf(training_df['ttr_ratio'])
training_df = training_df[~mask]

training_df.columns

training_df['sentiment_difference'] = training_df['response_b_sentiment'] - training_df['response_a_sentiment']

# prepare X and y values
X = training_df[['response_length_ratio','flesch_a', 'flesch_b','ttr_ratio','response_a_sentiment','response_b_sentiment','sentiment_difference','log_ttr_a', 'log_ttr_b','flesch_ratio']]
print(X['ttr_ratio'])

y = pd.factorize(training_df['winner'])[0] + 1
y_df = pd.DataFrame(y)
y_df.value_counts() # see how many wins, losses, ties

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## run random forest

rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=0)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# get accuracy measure

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(accuracy_rf)

# display a plot of one tree from random forest

single_tree = rf.estimators_[0]

plt.figure(figsize=(36, 10))
plot_tree(single_tree, 
          feature_names=X_train.columns, 
          class_names=['A', 'B', 'Tie'],
          filled=True,
          fontsize=10)
plt.show()

# list of feature importances

importances = rf.feature_importances_
feature_names = X_train.columns  # or pass them in manually if using NumPy array

pd.Series(importances, index=feature_names).sort_values()
plt.title("Feature Importances")
plt.show()

### ALL CODE BELOW NOT INCLUDED IN FINAL SUBMISSION
## just keeping for future reference playing with this code


## run gradient boosting

gbr_reg = GradientBoostingClassifier(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinks contribution of each tree
    random_state=42             # Ensures reproducibility
    )

gbr_reg.fit(X_train, y_train)
y_pred_gbr = gbr_reg.predict(X_test)

accuracy_gbr = accuracy_score(y_test, y_pred_gbr)
print(accuracy_gbr)

## run ADA boost

ada_reg = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),  # Base learner: shallow decision tree
    n_estimators=100,                              # Number of boosting rounds
    learning_rate=0.1,                             # Shrinks contribution of each learner
    random_state=42                                # Ensures reproducibility
    )

ada_reg.fit(X_train, y_train)
y_pred_ada = ada_reg.predict(X_test)

accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(accuracy_ada)

## Run Decision Tree

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)

# Predict using the Random Forest model.
y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test,y_pred_dt)
print(mse_dt)

## Multinomial Regression
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)

# Calculate residuals for the Multinomial Logistic Regression
y_pred_lr = lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(accuracy_lr)

mse_lr = mean_squared_error(y_test,y_pred_lr)
print(mse_lr)

log_loss_lr = log_loss(y_test, y_pred_lr)
print(log_loss_lr)

# Calculate residuals for the Logistic Regression
residuals_lr = y_test - y_pred_lr

# other metrics to measure model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

log_loss_2 = log_loss(y_test, y_pred_2)

### Sentence transformer method

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Combine A and B into one string to encode their relation
combined = training_df['response_a'] + ' [SEP] ' + train_data['response_b']

# Get embeddings
X = embedder.encode(combined, show_progress_bar=True)

# Your label: 0 = A wins, 1 = B wins, 2 = Tie
y = training_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values.argmax(axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["A Wins", "B Wins", "Tie"]))