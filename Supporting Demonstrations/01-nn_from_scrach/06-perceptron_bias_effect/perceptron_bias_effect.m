% ============================================================
% Perceptron Training: Effect of Bias on the Perceptron Model
% ============================================================
% Description:
% This code implements a single-layer perceptron in MATLAB to demonstrate
% the effect of bias on training. It compares two perceptrons: one with a bias term
% and one without. The model is trained on an AND gate problem.
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
%      of two inputs and applies a step function to predict the output (0 or 1).
%
% Problem: 
% This perceptron is trained to simulate a two-input AND gate. The program shows
% how including or excluding a bias affects the training outcome.
% ============================================================

% Define the input data for the AND gate
inputs = [0 0; 0 1; 1 0; 1 1];  % Each row is an input vector (2D)

% Define the target outputs for the AND gate
targets = [0; 0; 0; 1];  % Corresponding target values (binary outputs)

% Set the learning rate
learning_rate = 0.1;

% Set the number of epochs (training iterations)
epochs = 10;

% ============================================================
% Perceptron Training WITHOUT Bias
% ============================================================

fprintf('\nTraining Perceptron WITHOUT Bias:\n');

% Randomly initialize the weights for the perceptron without bias
weights_without_bias = rand(1, 2);

% Begin training loop for the given number of epochs
for epoch = 1:epochs
    fprintf('\nEpoch %d\n', epoch);
    for i = 1:size(inputs, 1)
        % Compute the net input (weighted sum of inputs WITHOUT bias)
        net_input = dot(inputs(i, :), weights_without_bias);

        % Activation function (step function), output is 1 if net_input > 0, otherwise 0
        output = net_input > 0;

        % Compute the error (difference between target and predicted output)
        error = targets(i) - output;

        % Update weights based on the error and learning rate
        weights_without_bias = weights_without_bias + learning_rate * error * inputs(i, :);

        % Display the current inputs, predicted output, error, and updated weights
        fprintf('Input: [%d %d], Target: %d, Predicted: %d, Error: %d\n', ...
                inputs(i, 1), inputs(i, 2), targets(i), output, error);
        fprintf('Updated Weights (No Bias): [%.4f, %.4f]\n', weights_without_bias(1), weights_without_bias(2));
    end
end

% Display final trained weights without bias
fprintf('\nFinal Trained Weights WITHOUT Bias: [%.4f, %.4f]\n', weights_without_bias(1), weights_without_bias(2));

% ============================================================
% Perceptron Training WITH Bias
% ============================================================

fprintf('\nTraining Perceptron WITH Bias:\n');

% Randomly initialize the weights and bias for the perceptron
weights_with_bias = rand(1, 2);
bias = rand;

% Begin training loop for the given number of epochs
for epoch = 1:epochs
    fprintf('\nEpoch %d\n', epoch);
    for i = 1:size(inputs, 1)
        % Compute the net input (weighted sum of inputs + bias)
        net_input = dot(inputs(i, :), weights_with_bias) + bias;

        % Activation function (step function), output is 1 if net_input > 0, otherwise 0
        output = net_input > 0;

        % Compute the error (difference between target and predicted output)
        error = targets(i) - output;

        % Update weights based on the error and learning rate
        weights_with_bias = weights_with_bias + learning_rate * error * inputs(i, :);

        % Update bias using the error and learning rate
        bias = bias + learning_rate * error;

        % Display the current inputs, predicted output, error, updated weights and bias
        fprintf('Input: [%d %d], Target: %d, Predicted: %d, Error: %d\n', ...
                inputs(i, 1), inputs(i, 2), targets(i), output, error);
        fprintf('Updated Weights (With Bias): [%.4f, %.4f], Updated Bias: %.4f\n', weights_with_bias(1), weights_with_bias(2), bias);
    end
end

% Display final trained weights and bias
fprintf('\nFinal Trained Weights WITH Bias: [%.4f, %.4f]\n', weights_with_bias(1), weights_with_bias(2));
fprintf('Final Trained Bias: %.4f\n', bias);

% ============================================================
% Model Testing Phase (With Bias)
% ============================================================

% Define the testing data (same as training data in this case)
test_inputs = [0 0; 0 1; 1 0; 1 1];  % Input patterns to test
test_targets = [0; 0; 0; 1];  % Expected outputs for the AND gate

% Test the perceptron (with bias) on each input in the test set
fprintf('\nTesting the trained perceptron WITH bias:\n');
for i = 1:size(test_inputs, 1)
    % Compute the net input using the trained weights and bias
    net_input = dot(test_inputs(i, :), weights_with_bias) + bias;

    % Activation function (step function)
    output = net_input > 0;

    % Display the test input, predicted output, and expected output
    fprintf('Test Input: [%d %d], Predicted: %d, Expected: %d\n', ...
            test_inputs(i, 1), test_inputs(i, 2), output, test_targets(i));
end
