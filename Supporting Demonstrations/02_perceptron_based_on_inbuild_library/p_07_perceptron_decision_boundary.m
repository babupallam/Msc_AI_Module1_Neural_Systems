% p_07_perceptron_decision_boundary.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Visualize the decision boundary of a trained perceptron.
% Purpose: Demonstrates how a perceptron creates a decision boundary after training.

% Define inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron and train it
net = perceptron;
net = train(net, inputs, targets);

% Plot decision boundary
figure;
plotpv(inputs, targets);
plotpc(net.IW{1}, net.b{1});
title('Perceptron Decision Boundary');
