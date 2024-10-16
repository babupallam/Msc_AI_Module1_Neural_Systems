% p_16_perceptron_regularization.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Introduce regularization to prevent overfitting.
% Purpose: Shows how regularization can help control overfitting in perceptron training.

% Define inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron
net = perceptron;

% Use regularization by modifying training parameters
net.performParam.regularization = 0.1; % Regularization factor

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display output
disp('Outputs with regularization:');
disp(output);
