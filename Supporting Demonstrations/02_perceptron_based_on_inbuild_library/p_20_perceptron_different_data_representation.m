% p_20_perceptron_different_data_representation.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Compare binary vs bipolar input representation in perceptron training.
% Purpose: Shows the effect of different input representations (binary vs bipolar) on perceptron performance.

% Define binary inputs and targets
inputs_bin = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Define bipolar inputs (-1 for 0, and 1 for 1)
inputs_bipolar = [-1 -1; -1 1; 1 -1; 1 1]';
targets_bipolar = [0 1 1 1];

% Create a perceptron for binary input
net_bin = perceptron;
net_bin = train(net_bin, inputs_bin, targets);

% Test the network
output_bin = net_bin(inputs_bin);

% Create a perceptron for bipolar input
net_bipolar = perceptron;
net_bipolar = train(net_bipolar, inputs_bipolar, targets_bipolar);

% Test the bipolar network
output_bipolar = net_bipolar(inputs_bipolar);

% Display output
disp('Outputs with binary inputs:');
disp(output_bin);

disp('Outputs with bipolar inputs:');
disp(output_bipolar);
