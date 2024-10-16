% p_01_perceptron_basic.m
% Model Architecture: Single-layer perceptron with 2 input neurons (for 2D points) and 1 output neuron.
% Task: Simple binary classification (OR gate).
% Purpose: Basic example to show perceptron training and classification.

% Define inputs (4 points in 2D space)
inputs = [0 0; 0 1; 1 0; 1 1]';

% Define targets (desired output for OR logic gate)
targets = [0 1 1 1];

% Create a perceptron (single-layer, 1 output neuron)
net = perceptron;

% Train the perceptron
net = train(net, inputs, targets);

% Test the perceptron with the same inputs
output = net(inputs);

% Display the output
disp('Outputs after training:');
disp(output);

% Visualize the results
figure;
plotpv(inputs, targets);
plotpc(net.IW{1}, net.b{1});
title('Perceptron Basic Example');
