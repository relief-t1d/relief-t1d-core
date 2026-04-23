import argparse
import sys

# ==========================================
# 1. MODEL PARAMETERS (Weights & Bias)
# ==========================================
# These values were extracted from the trained Keras Linear Model (.h5)
WEIGHTS = [
    0.043323018, -0.042692296, -0.06836648, 0.29572073,
    -0.87264186, 1.5070925, -2.7938073, 2.8550856
]
BIAS = 6.68784761428833
REQUIRED_LENGTH = len(WEIGHTS)


# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================
def predict_glucose(historical_readings):
    """
    Calculates the future glucose value based on 8 past readings.
    """
    if len(historical_readings) != REQUIRED_LENGTH:
        raise ValueError(
            f"Expected exactly {REQUIRED_LENGTH} historical readings, "
            f"but got {len(historical_readings)}."
        )

    # Linear equation: y = (w1*x1 + w2*x2 + ... + w8*x8) + b
    prediction = sum(w * x for w, x in zip(WEIGHTS, historical_readings)) + BIAS
    return round(prediction, 2)


# ==========================================
# 3. MAIN COMMAND LINE INTERFACE
# ==========================================
def main():
    # Setup argument parser for command line execution
    parser = argparse.ArgumentParser(
        description="Predict glucose levels 60 minutes into the future based on a 15-min interval history."
    )
    parser.add_argument(
        'readings',
        metavar='X',
        type=float,
        nargs='*',
        help='8 historical glucose readings separated by spaces (from oldest to newest)'
    )

    args = parser.parse_args()

    # If the user runs the script without arguments, run a default example
    if not args.readings:
        print("--- No arguments provided. Running default example ---")
        example_readings = [100.0, 102.0, 105.0, 110.0, 115.0, 118.0, 120.0, 122.0]

        print(f"Input history: {example_readings}")
        result = predict_glucose(example_readings)
        print(f"Predicted Glucose: {result} mg/dL")

        print("\nTip: You can pass your own values via command line like this:")
        print("python predictor.py 100 102 105 110 115 118 120 122")
        sys.exit(0)

    # Check if exactly 8 values were provided
    if len(args.readings) != REQUIRED_LENGTH:
        print(f"Error: You must provide exactly {REQUIRED_LENGTH} values.")
        print(f"You provided {len(args.readings)} values.")
        sys.exit(1)

    # Execute prediction with user provided arguments
    try:
        result = predict_glucose(args.readings)
        print(f"Predicted Glucose: {result} mg/dL")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()