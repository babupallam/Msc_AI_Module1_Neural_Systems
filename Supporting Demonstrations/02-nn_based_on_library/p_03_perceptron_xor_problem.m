% p_03_perceptron_xor_problem.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Attempt to solve the XOR problem.
% Purpose: Highlights the limitations of a single-layer perceptron on non-linearly separable data (XOR).

% Define XOR inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 0];

% Create a perceptron
net = perceptron;

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display output
disp('Perceptron output for XOR problem:');
disp(output);

% Visualize the perceptron error in solving XOR
figure;
plotpv(inputs, targets);
title('Perceptron XOR Problem');
