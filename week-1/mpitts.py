def mcculloch_pitts_neuron(inputs, weights, bias, threshold):
    """
    McCulloch-Pitts model of a neuron.

    Args:
    - inputs (list): Binary inputs to the neuron.
    - weights (list): Corresponding weights for the inputs.
    - bias (float): Bias term.
    - threshold (float): Threshold value.

    Returns:
    - int: Output of the neuron (0 or 1).
    """
    # Compute the weighted sum of inputs
    weighted_sum = sum(w * x for w, x in zip(weights, inputs))

    # Add bias term
    weighted_sum += bias

    # Compute the output of the neuron based on threshold
    if weighted_sum >= threshold:
        return 1
    else:
        return 0

# Example usage:
inputs = [1, 0, 1]  # Example binary inputs
weights = [0.5, -0.3, 0.8]  # Example weights
bias = 0.2  # Example bias term
threshold = 0.5  # Example threshold

output = mcculloch_pitts_neuron(inputs, weights, bias, threshold)
print("Output of the neuron:", output)
