"""
Custom loss functions and metrics for blood glucose prediction.
"""

import tensorflow as tf

# ==========================================
# 1. BASIC OPERATIONS & STANDARD LOSSES
# ==========================================

def sigmoid(x, center, slope):
    """
    Auxiliary sigmoid function used to build differentiable clinical penalties.
    It creates a smooth, continuous approximation of step boundaries (hard limits), 
    allowing gradients to flow properly during neural network backpropagation.
    """
    return 1.0 / (1.0 + tf.exp(-slope * (x - center)))

def MAE(y_true, y_pred):
    """Mean Absolute Error (MAE)."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    return tf.reduce_mean(tf.abs(y_pred - y_true))

# Mean Absolute Deviation (MAD) is mathematically equivalent to MAE in this context.
MAD = MAE 

def MARD(y_true, y_pred): 
    """
    Mean Absolute Relative Difference (MARD).
    Widely used in diabetology to evaluate Continuous Glucose Monitors (CGMs).
    It is mathematically equivalent to the Mean Absolute Percentage Error (MAPE) 
    used in general time-series forecasting.
    """
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    epsilon = tf.constant(1e-10, dtype=tf.float64)
    relative_diff = tf.abs(y_pred - y_true) / tf.maximum(tf.abs(y_true), epsilon)
    return tf.reduce_mean(relative_diff)

def MSE(y_true, y_pred):
    """Mean Squared Error (MSE)."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    return tf.square(y_pred - y_true)

def RMSE(y_true, y_pred):
    """Root Mean Squared Error (RMSE)."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# ==========================================
# 2. CLINICAL LOSS FUNCTIONS
# ==========================================

def clinical_penalty(y_true, y_pred):
    """
    Clinical penalty function for blood glucose prediction.
    Based on the paper: "A Glucose-Specific Metric to Assess Predictors and Identify Models" (Del Favero et al.) 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6135492
    https://github.com/MiceLab/MiceLab-Grammatical-Evolution-and-Type-1-Diabetes/blob/master/src/ge/EvaluateDataTable.java
    
    It outputs an asymmetrical penalty multiplier bounded in [1.0, 2.5]. 
    It strictly evaluates the severity of the error from a clinical standpoint, 
    penalizing missed hypoglycemia more heavily than benign hyperglycemia deviations.
    """
    output = tf.cast(y_pred, tf.float64)
    target = tf.cast(y_true, tf.float64)
    
    # Parameters defined by Del Favero et al.
    tetaH, betaH = 155.0, 100.0   # Hyperglycemia thresholds and transition widths
    tetaL, betaL = 85.0, 30.0     # Hypoglycemia thresholds and transition widths
    lambdaH, lambdaL = 20.0, 10.0 # Tolerance widths for prediction errors
    alphaL, alphaH = 1.5, 1.0     # Max penalty weights (Hypo is penalized more: alphaL > alphaH)

    # Zone 1 Penalty: Missed or underestimated Hypoglycemia
    Aux1zona1 = sigmoid(target, tetaL - (betaL / 2.0), -0.3)
    Aux2zona1 = sigmoid(output, target + (lambdaL / 2.0), 0.6)
    zona1 = alphaL * Aux1zona1 * Aux2zona1

    # Zone 2 Penalty: Missed or underestimated Hyperglycemia
    Aux1zona2 = sigmoid(target, tetaH + (betaH / 2.0), 0.1)
    Aux2zona2 = sigmoid(output, target - (lambdaH / 2.0), -0.4)
    zona2 = alphaH * Aux1zona2 * Aux2zona2
    
    # Base penalty is 1.0 (equivalent to a standard loss if no clinical risk is detected)
    return zona1 + zona2 + 1.0

def gMAD(y_true, y_pred):
    """
    Glucose-specific Mean Absolute Deviation (gMAD).
    Multiplies the absolute error by the Del Favero clinical penalty multiplier.
    """
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    return clinical_penalty(y_true, y_pred) * tf.abs(y_true - y_pred)

def gMSE(y_true, y_pred):
    """
    Glucose-specific Mean Squared Error (gMSE).
    Multiplies the squared error by the Del Favero clinical penalty multiplier.
    """
    return clinical_penalty(y_true, y_pred) * MSE(y_true, y_pred)

def c_sig(y_true, y_pred, p=0.1):
    """
    Continuous, differentiable approximation of the Clarke Error Grid (CEG) penalty.
    Uses sigmoids to softly evaluate boundaries for gradient backpropagation.
    """
    output = tf.cast(y_pred, tf.float64) # Y-Axis (Predictions)
    target = tf.cast(y_true, tf.float64) # X-Axis (Reference Values)
    
    # The 'p' parameter controls the steepness of the sigmoid boundaries.
    # p > 0 is used for upper/right bounds; p < 0 is used for lower/left bounds.
    
    ## ZONE A (Clinically accurate / Safe zone) ################################
    A = 1.0
    
    ## ZONE D1 (Failure to detect hypoglycemia) ################################
    # D11: Horizontal line between A and D1
    pD11 = p 
    # mD11 = 0.0 # Slope is 0.0 by definition
    cD11 = 70.0  # Intercept is 70.0 by definition
    D11 = sigmoid(output, cD11, pD11) 
    
    # D12: Diagonal line between A and D1
    pD12 = p 
    mD12 = 1.2   # Slope: (500-70)/(500/1.2 - 175/3) = 6/5 = 1.2 (+20% error boundary)
    cD12 = 0.0   # Intercept: 70.0 - mD12 * 175.0/3.0 = 0.0 
    D12 = sigmoid(output - mD12 * target, cD12, pD12)
    
    # D13: Vertical line between D1 and B1 or C1
    pD13 = -p 
    # mD13 = 1.0 
    cD13 = 70.0  # Intercept is 70.0 by definition
    D13 = sigmoid(target, cD13, pD13)
    
    D1 = D11 * D12 * D13 * 2.0
    
    ## ZONE E1 (Erroneous treatment in hypoglycemia) ###########################
    # Explicit formulation is not strictly needed here as it implicitly 
    # emerges from the overlap of D1 and C1.
    pE11 = p 
    cE11 = 180.0 # Intercept is 180.0 by definition
    E11 = sigmoid(output, cE11, pE11)
    
    E1 = 0.0
    
    ## ZONE C1 (Overcorrection in hypoglycemia) ################################
    # C11: Line between C1 and D1
    C11 = E11
    
    # C12: Diagonal line between C1 and B1
    pC12 = p 
    # mC12 = 1.0 # Slope: (500-180)/(390-70) = 1.0
    cC12 = 110.0 # Intercept: 180 - mC12 * 70 = 110.0
    C12 = sigmoid(output - target, cC12, pC12)
    
    C1 = C11 * C12

    ## ZONE B1 (Benign errors, predicting higher than actual) ##################
    # B11: Horizontal line between A and B1
    B11 = D11 
    # B12: Diagonal line between A and B1
    B12 = D12 # (+20% error boundary)
    
    B1 = B11 * B12
    
    ## ZONE B2 (Benign errors, predicting lower than actual) ###################
    # B21: Vertical line between A and B2
    pB21 = p 
    # mB21 = 1.0 # By definition
    cB21 = 70.0  # By definition
    B21 = sigmoid(target, cB21, pB21)
    
    # B22: Diagonal line between A and B2
    pB22 = -p 
    mB22 = 0.8   # Slope: (400-56)/(500-70) = 4/5 = 0.8 (-20% error boundary)
    cB22 = 0.0   # Intercept: 56 - mB22 * 70 = 0.0
    B22 = sigmoid(output - mB22 * target, cB22, pB22)
    
    B2 = B21 * B22
    
    ## ZONE C2 (Overcorrection in hyperglycemia) ###############################
    # C21: Diagonal line between B2 and C2
    pC21 = -p 
    mC21 = 1.4   # Slope: 70/(180-130) = 7/5 = 1.4
    cC21 = -182.0# Intercept: (0-70)/(180-130) * 130.0 = -182.0
    C21 = sigmoid(output - mC21 * target, cC21, pC21)
    
    # C22: Horizontal line between E2 and D2
    pC22 = -p 
    cC22 = 70.0  # By definition
    C22 = sigmoid(output, cC22, pC22)
    
    C2 = C21 * C22
    
    ## ZONE D2 (Failure to detect hyperglycemia) ###############################
    # D21: Vertical line between B2 and D2
    pD21 = p 
    cD21 = 240.0 # By definition
    D21 = sigmoid(target, cD21, pD21)
    
    # D22: Horizontal line between B2 and D2
    pD22 = -p 
    cD22 = 180.0 # By definition
    D22 = sigmoid(output, cD22, pD22)
    
    D2 = D21 * D22 * 2.0
    
    ## ZONE E2 (Erroneous treatment in hyperglycemia) ##########################
    # E21: Vertical line between C2 and E2
    pE21 = p 
    cE21 = 180.0 # By definition
    E21 = sigmoid(target, cE21, pE21)
    
    # E22: Horizontal line between B2 or D2 and E2
    E22 = C22
    
    E2 = E21 * E22 * 2.0 * sigmoid(target, cD21, -pD21)
    
    # Compute the final penalty multiplier
    return A + B1 + B2 + C1 + C2 + D1 + D2 + E1 + E2

# Wrappers for c_sig with different steepness parameters (p)
def c_sig_p01(y_true, y_pred): 
    return c_sig(y_true, y_pred, p=0.1)
    
def c_sig_p06(y_true, y_pred): 
    return c_sig(y_true, y_pred, p=0.6)

# Hybrid metrics multiplying the CEG penalty multiplier by the MSE
def c_sig_p01_MSE(y_true, y_pred): 
    return c_sig(y_true, y_pred, p=0.1) * MSE(y_true, y_pred)
    
def c_sig_p06_MSE(y_true, y_pred): 
    return c_sig(y_true, y_pred, p=0.6) * MSE(y_true, y_pred)


# ==========================================
# 3. KERAS CUSTOM LOSS CLASSES
# ==========================================
# Wrapping the custom functions into tf.keras.losses.Loss classes allows them 
# to be properly instantiated, tracked, and serialized when saving the models.

class ClinicalPenaltyLoss(tf.keras.losses.Loss):
    """Keras Loss class wrapper for the Del Favero Clinical Penalty."""
    def __init__(self, name='Clinical_Penalty', **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        return clinical_penalty(y_true, y_pred)

class CSigP01Loss(tf.keras.losses.Loss):
    """Keras Loss class wrapper for continuous CEG with p=0.1."""
    def __init__(self, name='c_sig_p01', **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        return c_sig_p01(y_true, y_pred)
    
class CSigP06Loss(tf.keras.losses.Loss):
    """Keras Loss class wrapper for continuous CEG with p=0.6."""
    def __init__(self, name='c_sig_p06', **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        return c_sig_p06(y_true, y_pred)

# ==========================================
# 4. REGISTRY DICTIONARIES
# ==========================================

# Dictionary mapping string aliases to the actual loss/metric functions.
# Very useful for dynamically selecting loss functions via configuration files.
loss_functions_dic = {
    "CP": clinical_penalty,
    "cSigP01": c_sig_p01,
    "cSigP06": c_sig_p06,
    "RMSE": RMSE,
    "gMSE": gMSE,
    "gMAD": gMAD,
    "cSigP01MSE": c_sig_p01_MSE,
    "cSigP06MSE": c_sig_p06_MSE,
}

loss_function_name_list = list(loss_functions_dic.keys())