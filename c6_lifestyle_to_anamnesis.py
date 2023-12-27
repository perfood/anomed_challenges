import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def build_features(df_glucose, df_meals, df_anamnesis):
    glucose_features = []
    meal_features = []
    anamnesis_features = []
    user_ids = df_glucose["user_id"].unique()
    for user_id in user_ids:
        df_glucose_user = df_glucose[df_glucose["user_id"] == user_id]
        quantiles = df_glucose_user["bg"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        quantiles.index = ['q0.1', 'q0.25', 'q0.5', 'q0.75', 'q0.9']
        glucose_features.append(quantiles)
        # Extract macronutrient averages and change of delta_max with macronutrients
        df_meals_user = df_meals[df_meals["user_id"] == user_id]
        macro_means = df_meals_user[["carbohydrate_g", "protein_g", "fat_g", "fiber_g"]].mean()
        macro_means.index = ["carbohydrate_g_mean", "protein_g_mean", "fat_g_mean", "fiber_g_mean"]
        macro_delta_corr = df_meals_user[["carbohydrate_g", "protein_g", "fat_g", "fiber_g", "delta_max"]].corr()[
                               "delta_max"][:-1]
        macro_delta_corr.index = ["carbohydrate_corr", "protein_corr", "fat_corr",
                                  "fiber_corr"]
        meal_features.append(pd.concat([macro_means, macro_delta_corr]))
        # Extract anamnesis features
        df_anamnesis_user = df_anamnesis[df_anamnesis["user_id"] == user_id]
        anamnesis_features.append(df_anamnesis_user.iloc[0, 1:])

    glucose_features = pd.concat(glucose_features, axis=1).transpose()
    meal_features = pd.concat(meal_features, axis=1).transpose()
    anamnesis_features = pd.concat(anamnesis_features, axis=1).transpose()

    return glucose_features, meal_features, anamnesis_features


df_meals_train = pd.read_csv("data/c6_meals_train.csv")
df_meals_train["start_time"] = pd.to_datetime(df_meals_train["start_time"])
df_meals_test = pd.read_csv("data/c6_meals_test.csv")
df_meals_test["start_time"] = pd.to_datetime(df_meals_test["start_time"])
df_anamnesis_train = pd.read_csv("data/c6_anamnesis_train.csv")
df_anamnesis_test = pd.read_csv("data/c6_anamnesis_test.csv")
df_glucose_train = pd.read_csv("data/c6_glucose_train.csv")
# df_glucose_train = pd.read_csv("data/c6_glucose_train.csv", nrows=100000)
df_glucose_train["time"] = pd.to_datetime(df_glucose_train["time"])
df_glucose_test = pd.read_csv("data/c6_glucose_test.csv")
# df_glucose_test = pd.read_csv("data/c6_glucose_test.csv", nrows=100000)
df_glucose_test["time"] = pd.to_datetime(df_glucose_test["time"])

glucose_features_train, meal_features_train, anamnesis_features_train = build_features(df_glucose_train, df_meals_train, df_anamnesis_train)
glucose_features_test, meal_features_test, anamnesis_features_test = build_features(df_glucose_test, df_meals_test, df_anamnesis_test)

X_train = pd.concat([glucose_features_train.reset_index(drop=True), meal_features_train.reset_index(drop=True)], axis=1)
X_test = pd.concat([glucose_features_test.reset_index(drop=True), meal_features_test.reset_index(drop=True)], axis=1)


# Train a model for each anamnesis feature
for feature in anamnesis_features_train.columns:
    y_train = anamnesis_features_train[feature]
    y_test = anamnesis_features_test[feature]

    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    rmse = mean_squared_error(y_test, predictions)**0.5
    print(f'Root Mean Squared Error for {feature}: {rmse}\n')

