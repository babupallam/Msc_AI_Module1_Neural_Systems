function output = neuron_model(input, weights, bias)
    % NEURON_MODEL: This function represents a single neuron.
    % The neuron computes a weighted sum of inputs, adds a bias, 
    % and applies an activation function (in this case, Sigmoid).
    %
    % Inputs:
    % input - A vector of input values (features)
    % weights - A vector of weights corresponding to the inputs
    % bias - A scalar bias term added to the weighted sum
    %
    % Output:
    % output - The result after applying the activation function
    
    % Step 1: Compute the weighted sum of inputs and add the bias
    z = dot(input, weights) + bias;  % Linear combination of inputs
    
    % Step 2: Pass the result through the sigmoid activation function
    output = sigmoid(z);  % Sigmoid activation function is applied
end

function y = sigmoid(x)
    % SIGMOID: This is a standard sigmoid activation function.
    % It maps input values into the range (0, 1), allowing non-linear transformations.
    %
    % Input:
    % x - Input value (can be scalar or vector)
    %
    % Output:
    % y - Sigmoid of the input value
    
    y = 1 ./ (1 + exp(-x));  % Sigmoid formula
end
