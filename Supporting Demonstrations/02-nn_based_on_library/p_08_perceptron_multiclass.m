% p_08_perceptron_multiclass.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron (multiclass classification).
% Task: Classify data into 3 classes using a perceptron.
% Purpose: Demonstrates multiclass classification with a perceptron.

% Define inputs for 3-class classification problem
inputs = [0 0; 0 1; 1 0; 1 1; 2 0; 2 1]';
targets = [1 1 2 2 3 3];

% Create a perceptron
net = perceptron;

% Train the perceptron
net = train(net, inputs, targets);

% Test the perceptron
output = net(inputs);

% Display output
disp('Outputs for multiclass classification:');
disp(output);

% Visualize results
figure;
plotpv(inputs, targets);
title('Perceptron Multiclass Classification');
