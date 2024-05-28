import math
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

userdf = pd.read_csv(r'C:\Users\illes\Downloads\users0901_0908.csv')

userdf['timestamp'] = pd.to_datetime(userdf['timestamp'], utc=True,format='mixed')
userdf['timestamp'] = userdf['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df= userdf.groupby(userdf['timestamp']).count()['ICAO Code']

#%%
from sklearn.neural_network import MLPRegressor

# Step 1: Prepare the data
X = df.iloc[:-1].values.reshape(-1, 1)
y = df.iloc[1:].values.ravel()

# Include lagged features
lags = [1, 2, 3, 4]  # Additional lag values
for lag in lags:
    X_lagged = np.roll(y, lag)
    X_lagged[0:lag] = 0  # Replace the first values with 0 or any appropriate value
    X = np.column_stack((X, X_lagged))

# Step 2: Split the data into training and testing sets
train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=65000,activation= 'relu',alpha= 0.01, learning_rate= 'constant',solver= 'adam',random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test_scaled)

# Step 6: Evaluate the model
# Shift y_test by 1
y_test_shifted = y_test

# Adjust predictions accordingly
predictions_shifted = predictions

# Calculate accuracy metrics with shifted y_test
mse = mean_squared_error(y_test_shifted, predictions_shifted)
r_squared = r2_score(y_test_shifted, predictions_shifted)
print("Mean Squared Error:", mse)
print("R-squared:", r_squared)

# Step 7: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size+1:train_size+len(y_test_shifted)+1], y_test_shifted, label='Actual')
plt.plot(df.index[train_size+1:train_size+len(predictions_shifted)+1], predictions_shifted, label='Predicted', color='red')
plt.xticks(rotation=90)
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
#%%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Prepare the data
X = df.iloc[:-1].values.reshape(-1, 1)
y = df.iloc[1:].values.ravel()

# Include lagged features
lags = [1, 2, 3, 4]  # Additional lag values
for lag in lags:
    X_lagged = np.roll(y, lag)
    X_lagged[0:lag] = 0  # Replace the first values with 0 or any appropriate value
    X = np.column_stack((X, X_lagged))

# Step 2: Split the data into training and testing sets
train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Perform hyperparameter optimization using Grid Search
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (200,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
}

grid_search = GridSearchCV(estimator=MLPRegressor(max_iter=65000, random_state=42),
                           param_grid=param_grid,
                           scoring='r2',
                           cv=5)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Step 5: Train the MLPRegressor with the best hyperparameters
best_model = MLPRegressor(max_iter=65000, random_state=42, **best_params)
best_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
predictions = best_model.predict(X_test_scaled)

# Step 7: Evaluate the model
# Shift y_test by 1
y_test_shifted = y_test

# Adjust predictions accordingly
predictions_shifted = predictions

# Calculate accuracy metrics with shifted y_test
mse = mean_squared_error(y_test_shifted, predictions_shifted)
r_squared = r2_score(y_test_shifted, predictions_shifted)
print("Mean Squared Error:", mse)
print("R-squared:", r_squared)

# Step 8: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size+1:train_size+len(y_test_shifted)+1], y_test_shifted, label='Actual')
plt.plot(df.index[train_size+1:train_size+len(predictions_shifted)+1], predictions_shifted, label='Predicted', color='red')
plt.xticks(rotation=90)
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
