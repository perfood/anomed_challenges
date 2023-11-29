import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


df_meals_train = pd.read_csv("data/c3_meals_train.csv")
df_meals_train["start_time"] = pd.to_datetime(df_meals_train["start_time"])
df_glucose_train = pd.read_csv("data/c3_glucose_train.csv")
df_glucose_train["time"] = pd.to_datetime(df_glucose_train["time"])
df_final_test_meal = pd.read_csv("data/c3_final_meal_test.csv")
df_final_test_meal["start_time"] = pd.to_datetime(df_final_test_meal["start_time"])
df_final_test_glucose = pd.read_csv("data/c3_final_glucose_test.csv")
df_final_test_glucose["time"] = pd.to_datetime(df_final_test_glucose["time"])
df_glucose_test = pd.read_csv("data/c3_glucose_test.csv")
df_glucose_test["time"] = pd.to_datetime(df_glucose_test["time"])
df_meals_test = pd.read_csv("data/c3_meals_test.csv")
df_meals_test["start_time"] = pd.to_datetime(df_meals_test["start_time"])

time_after_meal = pd.Timedelta("180 minutes")
time_points_needed_train = int(time_after_meal.seconds/60) + 1
time_points_needed_test = int(time_after_meal.seconds/60/60)*4

df_glucose_train_list_out = []
df_features_train = []
for user_id in df_meals_train["user_id"].unique():
    df_user_meals_train = df_meals_train[df_meals_train["user_id"] == user_id]
    df_user_glucose_train = df_glucose_train[df_glucose_train["user_id"] == user_id]
    start_time = df_user_glucose_train["time"].min()
    end_time = df_user_glucose_train["time"].max()
    time_cutoff = start_time + (end_time - start_time) * 2/3

    # take glucose until 2/3 of the way through for features
    df_gluc_features = df_user_glucose_train[df_user_glucose_train["time"] < time_cutoff]
    df_gluc_features = df_gluc_features["bg"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    df_gluc_features = df_gluc_features.rename({0.1: "bg_10th", 0.25: "bg_25th", 0.5: "bg_50th", 0.75: "bg_75th", 0.9: "bg_90th"})

    # use macronutrient averages as features up until 2/3 of the way through
    df_meals_features = df_user_meals_train[df_user_meals_train["start_time"] < time_cutoff]
    df_meals_features = df_meals_features[["carbohydrate_g", "fat_g", "protein_g", "fiber_g"]].mean()
    df_meals_features = df_meals_features.rename({"carbohydrate_g": "carbohydrate_g_mean", "fat_g": "fat_g_mean",
                                                  "protein_g": "protein_g_mean", "fiber_g": "fiber_g_mean"})
    if df_meals_features.isna().any():
        continue

    # for each meal in the last third, check if enough glucose data exists before and after to add it as data point
    df_glucose_last_third = df_user_glucose_train[df_user_glucose_train["time"] >= time_cutoff]
    df_meals_last_third = df_user_meals_train[df_user_meals_train["start_time"] >= time_cutoff]
    for index, row in df_meals_last_third.iterrows():
        # check if there are enough glucose data points before and after meal
        meal_time = row["start_time"]
        glucose_end = meal_time + time_after_meal

        df_glucose_meal_after = df_glucose_last_third[(df_glucose_last_third["time"] >= meal_time) & (df_glucose_last_third["time"] <= glucose_end)]
        if len(df_glucose_meal_after) >= time_points_needed_train:
            # resample before and after meal in 15 minute intervals
            df_glucose_meal_after = df_glucose_meal_after.set_index("time")
            shift = pd.Timedelta(minutes=meal_time.minute % 15, seconds=meal_time.second)
            df_glucose_meal_after.index = df_glucose_meal_after.index - shift
            df_glucose_meal_after = df_glucose_meal_after.resample("15T").mean()
            df_glucose_meal_after.index = df_glucose_meal_after.index + shift
            df_glucose_meal_after = df_glucose_meal_after.reset_index()
            df_glucose_train_list_out.append(df_glucose_meal_after.iloc[1:]["bg"].rename(user_id).reset_index(drop=True))

            # transform datetime of meal_time to minute count of the day
            meal_time = meal_time.hour * 60 + meal_time.minute
            time_encoding = np.array([np.sin(meal_time * 2 * np.pi / (24*60)), np.cos(meal_time * 2 * np.pi / (24*60))])
            time_encoding = pd.Series(time_encoding, index=["time_sin", "time_cos"])

            df_features_train.append(pd.concat((pd.Series(user_id, index=["user_id"]), df_gluc_features, df_meals_features, time_encoding,
                                                row[["fat_g", "carbohydrate_g", "protein_g", "fiber_g"]])))

df_glucose_train_out = pd.concat(df_glucose_train_list_out, axis=1).T
df_features_train = pd.concat(df_features_train, axis=1).T.set_index("user_id")

# extract features from test runs
df_glucose_test_list_out = []
df_features_test = []
for i, user_id in enumerate(df_final_test_meal["user_id"].unique()):
    df_user_meals_test = df_meals_test[df_meals_test["user_id"] == user_id]
    df_user_glucose_test = df_glucose_test[df_glucose_test["user_id"] == user_id]

    # take glucose until 2/3 of the way through for features
    df_gluc_features = df_user_glucose_test["bg"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    df_gluc_features = df_gluc_features.rename({0.1: "bg_10th", 0.25: "bg_25th", 0.5: "bg_50th", 0.75: "bg_75th", 0.9: "bg_90th"})

    # use macronutrient averages as features up until 2/3 of the way through
    df_meals_features = df_user_meals_test[["carbohydrate_g", "fat_g", "protein_g", "fiber_g"]].mean()
    df_meals_features = df_meals_features.rename({"carbohydrate_g": "carbohydrate_g_mean", "fat_g": "fat_g_mean",
                                                  "protein_g": "protein_g_mean", "fiber_g": "fiber_g_mean"})
    if df_meals_features.isna().any() or df_gluc_features.isna().any():
        continue

    user_meal = df_final_test_meal[df_final_test_meal["user_id"] == user_id]
    meal_time = user_meal["start_time"].iloc[0]

    df_glucose_meal_after = df_final_test_glucose[df_final_test_glucose["user_id"] == user_id]
    if len(df_glucose_meal_after) >= time_points_needed_test:
        df_glucose_test_list_out.append(df_glucose_meal_after["bg"].rename(user_id).reset_index(drop=True))

        meal_time = meal_time.hour * 60 + meal_time.minute
        time_encoding = np.array([np.sin(meal_time * 2 * np.pi / (24*60)), np.cos(meal_time * 2 * np.pi / (24*60))])
        time_encoding = pd.Series(time_encoding, index=["time_sin", "time_cos"])

        df_features_test.append(pd.concat((pd.Series(user_id, index=["user_id"]), df_gluc_features, df_meals_features, time_encoding,
                                           user_meal.iloc[0][["fat_g", "carbohydrate_g", "protein_g", "fiber_g"]])))

df_glucose_test_out = pd.concat(df_glucose_test_list_out, axis=1).T
df_features_test = pd.concat(df_features_test, axis=1).T.set_index("user_id")

# train model
features_train = np.asarray(df_features_train).astype(np.float32)
glucose_train_out = np.asarray(df_glucose_train_out).astype(np.float32)
features_test = np.asarray(df_features_test).astype(np.float32)
glucose_test_out = np.asarray(df_glucose_test_out).astype(np.float32)

X_train, X_val, Y_train, Y_val = train_test_split(features_train, glucose_train_out,
                                                  test_size=0.2, random_state=0)

inputs = Input(shape=(X_train.shape[1]))

x = Dense(32, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
x = Dense(Y_train.shape[1])(x)

model = Model(inputs=[inputs], outputs=[x])

model.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss="mse",
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error")],
)

val_baseline = model.evaluate(X_val, Y_val, verbose=0)[0]

cb_lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7)
cb_early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.01,
    patience=25,
    verbose=0,
    mode="auto",
    baseline=val_baseline,
    restore_best_weights=True,
)

model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=1000,
    callbacks=[cb_lr_reducer, cb_early_stopping],
    verbose=1,
)

test_loss, test_mae = model.evaluate(features_test, glucose_test_out, verbose=0)
print("Test Loss: ", test_loss)
print("Test MAE: ", test_mae)
