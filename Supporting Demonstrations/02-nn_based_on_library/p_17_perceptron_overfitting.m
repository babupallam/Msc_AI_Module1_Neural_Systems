% p_17_perceptron_overfitting.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Explore overfitting with a small dataset.
% Purpose: Demonstrates how a perceptron can overfit on small datasets and perform poorly on unseen data.

% Define a small set of inputs and targets
inputs = [0 0; 0 1; 1 0]';
targets = [0 1 1];

% Create a perceptron
net = perceptron;

% Train the network with the small dataset
net = train(net, inputs, targets);

% Test the network with unseen data
unseen_inputs = [1 1]';
output_unseen = net(unseen_inputs);

% Display output
disp('Output for unseen data (overfitting test):');
disp(output_unseen);
