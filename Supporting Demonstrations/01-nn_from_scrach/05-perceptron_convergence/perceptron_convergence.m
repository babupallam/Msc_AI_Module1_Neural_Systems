% ============================================================
% Perceptron Training with Error Convergence Plot
% ============================================================
% Description:
% This code implements a single-layer perceptron in MATLAB that plots the 
% convergence of error during training. It demonstrates how the total error 
% decreases over training epochs as the perceptron learns.
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
% This perceptron is trained to simulate a two-input AND gate and plots the error convergence 
% during the training process. The plot shows how the total error changes over the epochs.
% ============================================================

% Define the input data for the AND gate
inputs = [0 0; 0 1; 1 0; 1 1];  % Each row is an input vector (2D)

% Define the target outputs for the AND gate
targets = [0; 0; 0; 1];  % Corresponding target values (binary outputs)

% Randomly initialize the weights for the perceptron
weights = rand(1, 2);

% Randomly initialize the bias term
bias = rand;

% Set the learning rate (determines step size in weight adjustment)
learning_rate = 0.1;

% Set the number of epochs (training iterations)
epochs = 10;

% Initialize an array to store total error at each epoch
errors = zeros(epochs, 1);

% Begin training loop for the given number of epochs
for epoch = 1:epochs
    total_error = 0;  % Initialize total error for the current epoch
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

        % Sum absolute error for this epoch
        total_error = total_error + abs(error);

        % Display the current inputs, predicted output, error, and updated weights
        fprintf('Input: [%d %d], Target: %d, Predicted: %d, Error: %d\n', ...
                inputs(i, 1), inputs(i, 2), targets(i), output, error);
        fprintf('Updated Weights: [%.4f, %.4f], Updated Bias: %.4f\n', weights(1), weights(2), bias);
    end

    % Store the total error for the current epoch
    errors(epoch) = total_error;

    % Display the total error for the epoch
    fprintf('Total Error for Epoch %d: %.4f\n', epoch, total_error);
end

% After training, display the final trained weights and bias
fprintf('\nTraining complete.\n');
fprintf('Final Trained Weights: [%.4f, %.4f]\n', weights(1), weights(2));
fprintf('Final Trained Bias: %.4f\n', bias);

% ============================================================
% Plotting Error Convergence
% ============================================================

% Plot the total error for each epoch to visualize convergence
figure;
plot(1:epochs, errors, '-o');  % Plot error vs epochs
xlabel('Epochs');
ylabel('Total Error');
title('Error Convergence in Perceptron');
grid on;


% ============================================================
% Model Testing Phase
% ============================================================

% Define testing data (can be the same as training data)
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