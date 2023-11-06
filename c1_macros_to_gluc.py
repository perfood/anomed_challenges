import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


X_train = pd.read_csv("data/c1_X_train.csv")
Y_train = pd.read_csv("data/c1_Y_train.csv")
X_test = pd.read_csv("data/c1_X_test.csv")
Y_test = pd.read_csv("data/c1_Y_test.csv")

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

inputs = tf.keras.Input(shape=(X_train.shape[1]))
x = Dense(32, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
x = Dense(Y_train.shape[1])(x)

model = tf.keras.Model(
    inputs=[inputs],
    outputs=[x],
)

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

test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
print("Test Loss: ", test_loss)
print("Test MAE: ", test_mae)
