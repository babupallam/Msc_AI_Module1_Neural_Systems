% p_06_perceptron_convergence.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Observe convergence of perceptron training.
% Purpose: Demonstrates perceptron convergence through training iterations.

% Define inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron
net = perceptron;

% Track training progress
net.trainParam.showWindow = true; % Enable the training GUI

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display output
disp('Outputs after convergence:');
disp(output);
