def predict_fraud(model, input_df):
    """
    Returns binary prediction and fraud probability for the given input.
    """
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    return pred, proba
