% p_02_perceptron_logic_gate.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Learn the AND logic gate (linearly separable problem).
% Purpose: Demonstrates how a perceptron solves a linearly separable problem (AND gate).

% Define inputs (4 points for AND gate)
inputs = [0 0; 0 1; 1 0; 1 1]';

% Define targets (AND gate)
targets = [0 0 0 1];

% Create and train the perceptron
net = perceptron;
net = train(net, inputs, targets);

% Test the perceptron
output = net(inputs);

% Display output
disp('Perceptron output for AND gate:');
disp(output);

% Visualize the results
figure;
plotpv(inputs, targets);
plotpc(net.IW{1}, net.b{1});
title('Perceptron AND Gate');
