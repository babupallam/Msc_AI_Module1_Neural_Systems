% p_12_perceptron_weight_initialization.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Manual initialization of weights and biases, followed by training.
% Purpose: Demonstrates how manually setting the weights and bias impacts training.

% Define inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron
net = perceptron;

% Initialize the network with proper input and output sizes
net = configure(net, inputs, targets);

% Manually set the initial weights and bias
net.IW{1,1} = [1 -1]; % Initial random weights (1 row for 1 output neuron, 2 columns for 2 input neurons)
net.b{1} = 0.5;       % Initial bias

% Display initial weights and bias
disp('Initial weights:');
disp(net.IW{1,1});
disp('Initial bias:');
disp(net.b{1});

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display the final output after training
disp('Outputs after training with manual weight initialization:');
disp(output);
