import numpy as np
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model

# 1. Initial configuration
weights_path = "c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/Glucose_prediction_2026/models/cSigP01_Linear_H4_Fold1.weights.h5"  # <-- Change this to your actual .h5 file path
history_length = 8  # History length used in your script


# 2. Recreate the exact architecture of your LinearModel
def create_linear_model(history_len):
    i = Input(shape=(history_len,))
    x = Flatten()(i)
    x = Dense(1, activation=None)(x)
    model = Model(inputs=[i], outputs=[x])
    return model


# Initialize the model
model = create_linear_model(history_length)

# 3. Load the trained weights
try:
    model.load_weights(weights_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading weights. Please check the file path. Details: {e}")


# 4. model_prediction_scripts function
def predict_glucose(model, historical_readings):
    """
    historical_readings: list of glucose values (must contain exactly 'history_length' elements)
    """
    if len(historical_readings) != history_length:
        raise ValueError(f"Expected {history_length} values, but received {len(historical_readings)}.")

    # Keras expects data in batches.
    # We transform the list [x1, x2... x8] into an array of shape (1, 8)
    model_input = np.array(historical_readings).reshape(1, history_length)

    # Perform the prediction
    prediction = model.predict(model_input, verbose=0)

    # Extract the value from the resulting array
    return prediction[0][0]


# ==========================================
# USAGE EXAMPLE
# ==========================================
# 8 historical readings (every 15 mins, as indicated before)
# Order: from oldest to newest
my_readings = [110, 120, 140, 160, 180, 250, 251, 240]

result = predict_glucose(model, my_readings)
print(f"Input history: {my_readings}")
print(f"Future prediction: {result:.2f} mg/dL")