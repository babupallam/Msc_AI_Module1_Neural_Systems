% ============================================================
% Perceptron with Momentum
% ============================================================
% Description:
% This code implements a single-layer perceptron in MATLAB that uses
% momentum to speed up the learning process. Momentum helps the perceptron
% avoid local minima by adding a fraction of the previous weight update
% to the current one, allowing the model to maintain its direction of learning.
%
% Key Characteristics:
% 1. Type: Single-layer Perceptron with Momentum
%    - A basic form of a neural network with one output layer and no hidden layers.
%
% 2. Layers: 1 Layer
%    - The network consists of a single layer, which is the output layer.
%
% 3. Neurons: 1 Neuron
%    - The output layer has one neuron, which performs the weighted sum 
%      of two inputs, adds a bias, and applies a step function to predict the output (0 or 1).
%
% Momentum: 
% A momentum term is added to the weight and bias updates, allowing the perceptron
% to retain some information from the previous iteration. This can accelerate convergence
% and prevent oscillations around local minima.
% ============================================================

% Define the input data for the AND gate
inputs = [0 0; 0 1; 1 0; 1 1];  % Each row is an input vector (2D)

% Define the target outputs for the AND gate
targets = [0; 0; 0; 1];  % Corresponding target values (binary outputs)

% Randomly initialize the weights and bias
weights = rand(1, 2);
bias = rand;

% Set the learning rate
learning_rate = 0.1;

% Set the momentum factor (typically between 0 and 1)
momentum = 0.9;

% Set the number of epochs (training iterations)
epochs = 10;

% Initialize the previous weight and bias updates to zero (for momentum)
prev_weight_updates = zeros(1, 2);
prev_bias_update = 0;

% ============================================================
% Perceptron Training with Momentum
% ============================================================

% Begin training loop for the given number of epochs
for epoch = 1:epochs
    fprintf('\nEpoch %d\n', epoch);
    
    % Process each input sample
    for i = 1:size(inputs, 1)
        % Compute the net input (weighted sum of inputs + bias)
        net_input = dot(inputs(i, :), weights) + bias;

        % Activation function (step function), output is 1 if net_input > 0, otherwise 0
        output = net_input > 0;

        % Compute the error (difference between target and predicted output)
        error = targets(i) - output;

        % Calculate the weight updates with momentum
        weight_update = learning_rate * error * inputs(i, :) + momentum * prev_weight_updates;
        bias_update = learning_rate * error + momentum * prev_bias_update;

        % Update the weights and bias
        weights = weights + weight_update;
        bias = bias + bias_update;

        % Store the current weight and bias updates for the next iteration (momentum)
        prev_weight_updates = weight_update;
        prev_bias_update = bias_update;

        % Display the current inputs, predicted output, error, and updated weights/bias
        fprintf('Input: [%d %d], Target: %d, Predicted: %d, Error: %d\n', ...
                inputs(i, 1), inputs(i, 2), targets(i), output, error);
        fprintf('Updated Weights (With Momentum): [%.4f, %.4f], Updated Bias: %.4f\n', ...
                weights(1), weights(2), bias);
    end
end

% Display final trained weights and bias
fprintf('\nFinal Trained Weights: [%.4f, %.4f]\n', weights(1), weights(2));
fprintf('Final Trained Bias: %.4f\n', bias);

% ============================================================
% Model Testing Phase
% ============================================================

% Define the testing data (same as training data in this case)
test_inputs = [0 0; 0 1; 1 0; 1 1];  % Input patterns to test
test_targets = [0; 0; 0; 1];  % Expected outputs for the AND gate

% Test the trained perceptron on each input in the test set
fprintf('\nTesting the trained perceptron with momentum:\n');
for i = 1:size(test_inputs, 1)
    % Compute the net input using the trained weights and bias
    net_input = dot(test_inputs(i, :), weights) + bias;

    % Activation function (step function)
    output = net_input > 0;

    % Display the test input, predicted output, and expected output
    fprintf('Test Input: [%d %d], Predicted: %d, Expected: %d\n', ...
            test_inputs(i, 1), test_inputs(i, 2), output, test_targets(i));
end
