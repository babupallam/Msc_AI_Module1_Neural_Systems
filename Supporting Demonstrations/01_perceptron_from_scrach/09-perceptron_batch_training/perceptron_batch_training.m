% ============================================================
% Perceptron with Batch Training
% ============================================================
% Description:
% This code implements a single-layer perceptron in MATLAB using batch
% training. Unlike stochastic training (which updates weights after each
% input), batch training updates weights and bias after processing all
% the training inputs in an epoch.
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
% This perceptron is trained to simulate a two-input AND gate using batch training.
% In this approach, weight updates are accumulated for all inputs and applied
% only once after each epoch.
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

% Set the number of epochs (training iterations)
epochs = 10;

% ============================================================
% Batch Training
% ============================================================

% Begin training loop for the given number of epochs
for epoch = 1:epochs
    fprintf('\nEpoch %d\n', epoch);
    
    % Initialize cumulative weight and bias updates
    weight_updates = zeros(1, 2);
    bias_update = 0;
    
    % Process each input sample in the batch
    for i = 1:size(inputs, 1)
        % Compute the net input (weighted sum of inputs + bias)
        net_input = dot(inputs(i, :), weights) + bias;

        % Activation function (step function), output is 1 if net_input > 0, otherwise 0
        output = net_input > 0;

        % Compute the error (difference between target and predicted output)
        error = targets(i) - output;

        % Accumulate weight updates and bias updates based on the error and learning rate
        weight_updates = weight_updates + learning_rate * error * inputs(i, :);
        bias_update = bias_update + learning_rate * error;

        % Display the current inputs, predicted output, and error
        fprintf('Input: [%d %d], Target: %d, Predicted: %d, Error: %d\n', ...
                inputs(i, 1), inputs(i, 2), targets(i), output, error);
    end
    
    % Apply the accumulated updates after processing all inputs in the batch
    weights = weights + weight_updates;
    bias = bias + bias_update;

    % Display the updated weights and bias after the batch update
    fprintf('Updated Weights after Epoch %d: [%.4f, %.4f], Updated Bias: %.4f\n', epoch, weights(1), weights(2), bias);
end

% Display the final trained weights and bias
fprintf('\nTraining complete.\n');
fprintf('Final Trained Weights: [%.4f, %.4f]\n', weights(1), weights(2));
fprintf('Final Trained Bias: %.4f\n', bias);

% ============================================================
% Model Testing Phase
% ============================================================

% Define the testing data (same as training data in this case)
test_inputs = [0 0; 0 1; 1 0; 1 1];  % Input patterns to test
test_targets = [0; 0; 0; 1];  % Expected outputs for the AND gate

% Test the trained perceptron on each input in the test set
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
