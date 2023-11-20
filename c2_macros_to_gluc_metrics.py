import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


X_train = pd.read_csv("data/c2_X_train.csv")
Y_train = pd.read_csv("data/c2_Y_train.csv")
X_test = pd.read_csv("data/c2_X_test.csv")
Y_test = pd.read_csv("data/c2_Y_test.csv")

auc_train = Y_train["auc"]
delta_max_train = Y_train["delta_max"]
auc_test = Y_test["auc"]
delta_max_test = Y_test["delta_max"]

model_auc = RandomForestRegressor(n_estimators=100, random_state=0)

# Fit the model
model_auc.fit(X_train, auc_train)

# Predict on test data
test_predictions = model_auc.predict(X_test)
test_mae = mean_absolute_error(test_predictions, auc_test)
test_mse = mean_squared_error(test_predictions, auc_test)
print("Test AUC MAE: ", test_mae)
print("Test AUC MSE: ", test_mse)

model_d_max = RandomForestRegressor(n_estimators=100, random_state=0)

# Fit the model
model_d_max.fit(X_train, delta_max_train)

# Predict on test data
test_predictions = model_d_max.predict(X_test)
test_mae = mean_absolute_error(test_predictions, delta_max_test)
test_mse = mean_squared_error(test_predictions, delta_max_test)
print("Test delta_max MAE: ", test_mae)
print("Test delta_max MSE: ", test_mse)
