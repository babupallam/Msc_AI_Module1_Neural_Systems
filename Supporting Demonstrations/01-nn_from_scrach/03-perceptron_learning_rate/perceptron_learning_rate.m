% ============================================================
% Perceptron Training and Testing with Two Inputs
% ============================================================
% Description:
% This code implements a single-layer perceptron in MATLAB that has two input features.
% 
% Key Characteristics:
% 1. Type: Single-layer Perceptron
%    - A basic form of a neural network with only one output layer and no hidden layers.
%
% 2. Layers: 1 Layer
%    - The network consists of a single layer, which is the output layer.
%
% 3. Neurons: 1 Neuron
%    - The output layer has one neuron, which performs the weighted sum 
%      of two inputs, adds a bias, and applies a step function to predict the output (0 or 1).
%
% Problem: 
% This perceptron is trained to simulate a two-input AND gate using two input features and 
% then tested on unseen data to evaluate its performance.
% ============================================================

% Perceptron with Adjustable Learning Rate
% This script demonstrates the effect of learning rate on the training
% of a simple perceptron model for an AND gate problem.

% Define the input data for the AND gate
inputs = [0 0; 0 1; 1 0; 1 1];  % Each row is an input vector (2D)

% Define the target outputs for the AND gate
targets = [0; 0; 0; 1];  % Corresponding target values (binary outputs)

% Randomly initialize the weights for the perceptron
% Weights for the two input features
weights = rand(1, 2);

% Randomly initialize the bias term
bias = rand;

% Set the learning rate (a small value to explore its impact)
learning_rate = 0.01;

% Set the number of epochs (training iterations)
epochs = 10;

% Begin training loop for the given number of epochs
for epoch = 1:epochs
    fprintf('\nEpoch %d\n', epoch);  % Print the current epoch number
    for i = 1:size(inputs, 1)
        % Compute the net input (weighted sum of inputs + bias)
        net_input = dot(inputs(i, :), weights) + bias;

        % Activation function (step function), output is 1 if net_input > 0, otherwise 0
        output = net_input > 0;

        % Compute the error (difference between target and predicted output)
        error = targets(i) - output;

        % Update weights based on the error and learning rate
        weights = weights + learning_rate * error * inputs(i, :);

        % Update bias similarly using the error and learning rate
        bias = bias + learning_rate * error;

        % Display the current inputs, predicted output, error, and updated weights
        fprintf('Input: [%d %d], Target: %d, Predicted: %d, Error: %d\n', ...
                inputs(i, 1), inputs(i, 2), targets(i), output, error);
        fprintf('Updated Weights: [%.4f, %.4f], Updated Bias: %.4f\n', weights(1), weights(2), bias);
    end
end

% After training, display the final trained weights and bias
fprintf('\nTraining complete.\n');
fprintf('Final Trained Weights: [%.4f, %.4f]\n', weights(1), weights(2));
fprintf('Final Trained Bias: %.4f\n', bias);

% ============================================================
% Model Testing Phase
% ============================================================

% Testing data (can be same as training data or unseen data for generalization)
test_inputs = [0 0; 0 1; 1 0; 1 1];  % Input patterns to test
test_targets = [0; 0; 0; 1];  % Expected outputs for the AND gate

% Test the perceptron on each input in the test set
fprintf('\nTesting the trained perceptron:\n');
for i = 1:size(test_inputs, 1)
    % Compute the net input using the trained weights and bias
    net_input = dot(test_inputs(i, :), weights) + bias;

    % Activation function (step function)
    output = net_input > 0;

    % Display the test input, predicted output, and expected output
    fprintf('Test Input: [%d %d], Predicted: %d, Expected: %d\n', ...
            test_inputs(i, 1), test_inputs(i, 2), output, test_targets(i));
end
