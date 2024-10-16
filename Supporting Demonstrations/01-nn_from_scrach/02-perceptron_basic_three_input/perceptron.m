% ============================================================
% Perceptron Training and Testing with Three Inputs
% ============================================================
% Description:
% This code implements a single-layer perceptron in MATLAB that has three input features.
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
%      of three inputs, adds a bias, and applies a step function to predict the output (0 or 1).
%
% Problem: 
% This perceptron is trained to simulate a three-input AND gate using three input features and 
% then tested on unseen data to evaluate its performance.
% ============================================================

% Step 1: Define the training data for a three-input AND gate.
train_inputs = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];  
train_targets = [0; 0; 0; 0; 0; 0; 0; 1];  % Expected output for the three-input AND gate

% Step 2: Define the test data (same as training data in this case, but you can use new data).
test_inputs = train_inputs;  % Testing on the same inputs for simplicity
test_targets = train_targets;  % Expected outputs for the test set (AND gate)

% Step 3: Initialize weights and bias for three inputs.
weights = rand(1, 3);  % Random initialization of weights for three input features
bias = rand;           % Random initialization of bias
learning_rate = 0.1;   % Learning rate controls how much weights are adjusted in each step
epochs = 10;           % Number of training iterations over the dataset

% Display initial weights and bias
disp('Initial Weights and Bias:');
disp(weights);  % Display initial random weights
disp(bias);     % Display initial random bias

% Step 4: Training loop - Train the perceptron by adjusting weights and bias over multiple epochs.
for epoch = 1:epochs
    disp(['Epoch ', num2str(epoch), ':']);
    for i = 1:size(train_inputs, 1)
        % Compute the weighted sum (net input)
        net_input = dot(train_inputs(i, :), weights) + bias;
        
        % Apply the step activation function: output = 1 if net_input > 0, else output = 0
        output = net_input > 0;
        
        % Calculate the error (difference between expected target and perceptron output)
        error = train_targets(i) - output;
        
        % Update weights using the error and learning rate
        weights = weights + learning_rate * error * train_inputs(i, :);
        
        % Update bias using the error and learning rate
        bias = bias + learning_rate * error;
        
        % Display the updated weights and bias after each training sample
        disp(['  Sample ', num2str(i), ': Updated Weights = ', num2str(weights), ', Updated Bias = ', num2str(bias)]);
    end
end

% Step 5: Display the final trained weights and bias after training is complete.
disp('Final Trained Weights and Bias after training:');
disp(weights);
disp(bias);

% Step 6: Testing the trained perceptron on test data.
disp('Testing the trained perceptron on new data:');
for i = 1:size(test_inputs, 1)
    % Compute the net input for each test sample
    net_input = dot(test_inputs(i, :), weights) + bias;
    
    % Apply the step function to determine the perceptron's predicted output
    output = net_input > 0;
    
    % Display the test input, predicted output, and expected target output
    disp(['  Test Input: ', num2str(test_inputs(i, :)), ...
          ' => Predicted: ', num2str(output), ...
          ', Expected: ', num2str(test_targets(i))]);
end
