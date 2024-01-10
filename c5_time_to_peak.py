import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler


X_train = pd.read_csv("data/c5_X_train.csv")
Y_train = pd.read_csv("data/c5_Y_train.csv")
X_test = pd.read_csv("data/c5_X_test.csv")
Y_test = pd.read_csv("data/c5_Y_test.csv")

# create validation data
splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=0)
split = splitter.split(X_train, groups=X_train["user_id"])
train_indxs, valid_indxs = next(split)
X_valid = X_train.iloc[valid_indxs].copy()
X_train = X_train.iloc[train_indxs].copy()

Y_valid = Y_train.iloc[valid_indxs]
Y_train = Y_train.iloc[train_indxs]

X_train.drop(labels="user_id", axis=1, inplace=True)
X_valid.drop(labels="user_id", axis=1, inplace=True)
X_test.drop(labels="user_id", axis=1, inplace=True)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

tf.random.set_seed(0)
input = Input(shape=X_train.shape[1])
x = Dense(32, activation="relu")(input)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
output = Dense(Y_train.shape[1])(x)

model = tf.keras.Model(inputs=input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.1),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
)

val_baseline = model.evaluate(X_valid, Y_valid, verbose=0)[0]

cb_lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-10)

cb_early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=120,
    verbose=0,
    mode="auto",
    baseline=val_baseline,
    restore_best_weights=True,
)

model.fit(
    X_train,
    Y_train,
    validation_data=(X_valid, Y_valid),
    epochs=1000,
    callbacks=[cb_lr_reducer, cb_early_stopping],
    verbose=0,
)

test_mse, test_mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test: \n\t MSE: {test_mse} \n\t MAE: {test_mae}")
