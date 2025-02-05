import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True,
                   input_shape=input_shape,
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    return history

def predict_value(model, scaler, input_sequence, features):
    import numpy as np
    input_seq = np.array(input_sequence)
    scaled_seq = scaler.transform(input_seq)
    scaled_seq = np.expand_dims(scaled_seq, axis=0)
    prediction = model.predict(scaled_seq)
    zeros = np.zeros((prediction.shape[0], len(features) - 1))
    prediction_extended = np.hstack((prediction, zeros))
    predicted_close = scaler.inverse_transform(prediction_extended)[:, 0]
    return predicted_close[0]
