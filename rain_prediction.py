import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load the data
df = pd.read_csv(r'C:\Users\illes\Desktop\Időjárás adat\weather_calc.csv')
df.loc[df['precip'] != 0, 'precip'] = 1

# Pivot the DataFrame
pivot_df = df.pivot_table(index='datetime', columns='name', values='precip', aggfunc='first')

# If there are missing values after pivoting, you can fill them with 0
pivot_df = pivot_df.fillna(0)
#%%
# Select 'Fawley' as the target variable
target = 'Fawley'
y = pivot_df[target]
pivot_df=pivot_df.drop(columns=target)
# Create lagged values for each city
lags = [1, 2, 3, 4]
for lag in lags:
    for city in pivot_df.columns:
        pivot_df[f'{city}_lag{lag}'] = pivot_df[city].shift(lag)

# Remove rows with NaN values resulting from the shift
pivot_df = pivot_df.fillna(0)

# Prepare the data for training and testing
X = pivot_df

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
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
    'max_iter': [500, 1000]
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
#%%evaluation metrics
# Evaluate Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:")
print(rf_conf_matrix)
print("Random Forest F1 Score:", rf_f1)

# Evaluate MLP Neural Network model
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
mlp_conf_matrix = confusion_matrix(y_test, mlp_predictions)
mlp_f1 = f1_score(y_test, mlp_predictions, average='weighted')
print("\nMLP Neural Network Accuracy:", mlp_accuracy)
print("MLP Neural Network Confusion Matrix:")
print(mlp_conf_matrix)
print("MLP Neural Network F1 Score:", mlp_f1)

# Evaluate XGBoost model
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions, average='weighted')
print("\nXGBoost Accuracy:", xgb_accuracy)
print("XGBoost Confusion Matrix:")
print(xgb_conf_matrix)
print("XGBoost F1 Score:", xgb_f1)
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrices for Random Forest, MLP Neural Network, and XGBoost models
plt.figure(figsize=(15, 5))

# Plot Random Forest Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(rf_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')

# Plot MLP Neural Network Confusion Matrix
plt.subplot(1, 3, 2)
sns.heatmap(mlp_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('MLP Neural Network Confusion Matrix')

# Plot XGBoost Confusion Matrix
plt.subplot(1, 3, 3)
sns.heatmap(xgb_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('XGBoost Confusion Matrix')

plt.tight_layout()
plt.show()

#%%
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve and AUC for Random Forest
rf_probs = rf_best_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label='Random Forest (AUC = %0.2f)' % rf_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
#%%
# Get feature importances from the Random Forest model
feature_importance_rf = rf_best_model.feature_importances_

# Get the names of the features
feature_names = X.columns

# Sort feature importances in descending order and select the top 5
top_indices = np.argsort(feature_importance_rf)[::-1][:5]

# Plot top 5 feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(5), feature_importance_rf[top_indices], align='center')
plt.xticks(range(5), feature_names[top_indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top 5 Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

