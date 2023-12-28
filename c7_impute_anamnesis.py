import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.dummy import DummyRegressor, DummyClassifier

df_missing = pd.read_csv('data/c7_test_missing.csv').drop(columns=['user_id'])
df_complete = pd.read_csv('data/c7_test_complete.csv').drop(columns=['user_id'])
df_train = pd.read_csv('data/c7_train.csv').drop(columns=['user_id'])

test_scores = {}
baseline_scores = {}
for feature in df_missing.columns:
    X_train = df_train.drop(columns=[feature])
    y_train = df_train[feature]

    missing_indices = df_missing[df_missing[feature].isnull()].index
    predict_data = df_missing.loc[missing_indices]

    if predict_data.empty:
        continue

    if feature == "sex":
        model = RandomForestClassifier(random_state=0)
        baseline_model = DummyClassifier(strategy="most_frequent")
        score_function = accuracy_score
    else:
        model = RandomForestRegressor(random_state=0)
        baseline_model = DummyRegressor(strategy="mean")
        score_function = mean_squared_error

    model.fit(X_train, y_train)
    baseline_model.fit(X_train, y_train)

    X_predict = predict_data.drop(columns=[feature])
    predicted_values = model.predict(X_predict)
    baseline_predicted_values = baseline_model.predict(X_predict)

    df_missing.loc[missing_indices, feature] = predicted_values
    y_true = df_complete.loc[missing_indices, feature]
    y_pred = df_missing.loc[missing_indices, feature]
    if feature == "sex":
        test_score = score_function(y_true, y_pred)
        baseline_score = score_function(y_true, baseline_predicted_values)
    else:
        test_score = score_function(y_true, y_pred, squared=False)  # RMSE
        baseline_score = score_function(y_true, baseline_predicted_values, squared=False)

    test_scores[feature] = test_score
    baseline_scores[feature] = baseline_score

for feature, score in test_scores.items():
    baseline_score = baseline_scores[feature]
    if feature == "sex":
        print(f"Accuracy for {feature}: {score}")
        print(f"Baseline accuracy for {feature}: {baseline_score}")
    else:
        print(f"Root Mean Squared Error for {feature}: {score}")
        print(f"Baseline RMSE for {feature}: {baseline_score}")
