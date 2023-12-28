import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


df_missing = pd.read_csv('data/c7_test_missing.csv').drop(columns=['user_id'])
df_complete = pd.read_csv('data/c7_test_complete.csv').drop(columns=['user_id'])
df_train = pd.read_csv('data/c7_train.csv').drop(columns=['user_id'])

test_scores = {}
for feature in df_missing.columns:
    X_train = df_train.drop(columns=[feature])
    y_train = df_train[feature]

    missing_indices = df_missing[df_missing[feature].isnull()].index
    predict_data = df_missing.loc[missing_indices]

    if predict_data.empty:
        continue

    if feature == "sex":
        model = RandomForestClassifier(random_state=0)
        score_function = accuracy_score
    else:
        model = RandomForestRegressor(random_state=0)
        score_function = mean_squared_error

    model.fit(X_train, y_train)
    X_predict = predict_data.drop(columns=[feature])
    predicted_values = model.predict(X_predict)

    df_missing.loc[missing_indices, feature] = predicted_values
    y_true = df_complete.loc[missing_indices, feature]
    y_pred = df_missing.loc[missing_indices, feature]
    if feature == "sex":
        test_score = score_function(y_true, y_pred)
    else:
        test_score = score_function(y_true, y_pred, squared=False)  # RMSE

    test_scores[feature] = test_score

for feature, score in test_scores.items():
    if feature == "sex":
        print(f"Accuracy for {feature}: {score}")
    else:
        print(f"Root Mean Squared Error for {feature}: {score}")
