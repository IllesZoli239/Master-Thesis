import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r'C:\Users\illes\Desktop\Időjárás adat\weather_calc.csv')
df.loc[df['precip'] != 0, 'precip'] = 1

# Pivot the DataFrame
pivot_df = df.pivot_table(index='datetime', columns='name', values='precip', aggfunc='first')

# If there are missing values after pivoting, you can fill them with 0
pivot_df = pivot_df.fillna(0)

# Prepare the data for training and testing
X = pivot_df.dropna()
towns = X.columns
#%%
# Initialize lists to store accuracy and F1 scores for each town
accuracy_scores = []
f1_scores = []

# Loop over each town
for town in towns:
    print(f"Processing {town}")
    y = pivot_df[town]
    X = pivot_df.drop(columns=town)
    
    lags = [1, 2, 3, 4]
    for lag in lags:
        for city in X.columns:
            X[f'{city}_lag{lag}'] = X[city].shift(lag)
            
    X = X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grids for hyperparameter optimization
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    mlp_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (200,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'max_iter': [1500, 2000]
    }

    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Hyperparameter optimization using GridSearchCV
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='f1')
    mlp_grid_search = GridSearchCV(MLPClassifier(random_state=42), mlp_param_grid, cv=5, scoring='f1')
    xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=5, scoring='f1')

    # Fit the models
    rf_grid_search.fit(X_train, y_train)
    mlp_grid_search.fit(X_train, y_train)
    xgb_grid_search.fit(X_train, y_train)

    # Select the best models
    rf_best_model = rf_grid_search.best_estimator_
    mlp_best_model = mlp_grid_search.best_estimator_
    xgb_best_model = xgb_grid_search.best_estimator_

    # Make predictions
    rf_predictions = rf_best_model.predict(X_test)
    mlp_predictions = mlp_best_model.predict(X_test)
    xgb_predictions = xgb_best_model.predict(X_test)

    # Calculate accuracy and F1 score for each model
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_f1 = f1_score(y_test, rf_predictions)

    mlp_accuracy = accuracy_score(y_test, mlp_predictions)
    mlp_f1 = f1_score(y_test, mlp_predictions)

    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    xgb_f1 = f1_score(y_test, xgb_predictions)

    # Append accuracy and F1 scores to the lists
    accuracy_scores.append([rf_accuracy, mlp_accuracy, xgb_accuracy])
    f1_scores.append([rf_f1, mlp_f1, xgb_f1])
#%%
# Convert lists to NumPy arrays
accuracy_scores = np.array(accuracy_scores)
f1_scores = np.array(f1_scores)

# Plot accuracy scores
plt.figure(figsize=(10, 6))
plt.boxplot(accuracy_scores, labels=['Random Forest', 'MLP Neural Network', 'XGBoost'])
plt.title('Accuracy Scores for Different Towns')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()

# Plot F1 scores
plt.figure(figsize=(10, 6))
plt.boxplot(f1_scores, labels=['Random Forest', 'MLP Neural Network', 'XGBoost'])
plt.title('F1 Scores for Different Towns')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.show()
#%%
town_names = list(towns)

# Get the highest accuracy and F1 score for each town
max_accuracy = np.max(accuracy_scores, axis=1)
max_f1_score = np.max(f1_scores, axis=1)

# Plot accuracy and F1 scores for each town
plt.figure(figsize=(20, 8))

# Plot accuracy
plt.bar(np.arange(len(town_names)) - 0.2, max_accuracy, width=0.4, label='Accuracy')
# Plot F1 score
plt.bar(np.arange(len(town_names)) + 0.2, max_f1_score, width=0.4, label='F1 Score')

plt.xticks(np.arange(len(town_names)), town_names, rotation=45)
plt.xlabel('Town')
plt.ylabel('Score')
plt.title('Highest Accuracy and F1 Score for Different Towns')
plt.legend()
plt.tight_layout()
plt.show()
