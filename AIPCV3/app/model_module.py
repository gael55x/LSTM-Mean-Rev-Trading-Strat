import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
    
def build_model(input_shape):
    # we define a sequential model using lstm layers for time-series forecasting
    model = Sequential()
    # we add the first lstm layer with l2 regularization to reduce overfitting
    model.add(LSTM(units=128, return_sequences=True,
                   input_shape=input_shape,
                   kernel_regularizer=l2(0.001)))
    # we add dropout to further reduce overfitting
    model.add(Dropout(0.3))
    # we add a second lstm layer with fewer units, also with l2 regularization
    model.add(LSTM(units=64, return_sequences=False, kernel_regularizer=l2(0.001)))
    # we add dropout again
    model.add(Dropout(0.3))
    # we add a dense layer with a single unit for the final output (predicted close price)
    model.add(Dense(1))
    # we compile the model using adam optimizer and mse loss
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    # we use early stopping to stop training if val_loss stops improving
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    # we fit the model on training data and validate on test data
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
    # we convert the input sequence to a numpy array
    input_seq = np.array(input_sequence)
    # we scale the sequence using our stored scaler
    scaled_seq = scaler.transform(input_seq)
    # we add an extra dimension to match lstm's expected input shape
    scaled_seq = np.expand_dims(scaled_seq, axis=0)
    # we make the prediction
    prediction = model.predict(scaled_seq)
    # we pad the prediction with zeros for inverse transform, since we only predict 1 feature
    zeros = np.zeros((prediction.shape[0], len(features) - 1))
    prediction_extended = np.hstack((prediction, zeros))
    # we reverse the scaling to get the real predicted close price
    predicted_close = scaler.inverse_transform(prediction_extended)[:, 0]
    return predicted_close[0]
