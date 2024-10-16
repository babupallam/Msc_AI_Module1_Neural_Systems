% p_19_perceptron_hidden_layer_simulation.m
% Model Architecture: Multilayer network (simulated hidden layer).
% Task: Add a hidden layer to solve the XOR problem (which a single-layer perceptron cannot solve).
% Purpose: Demonstrates how adding hidden layers allows the model to solve non-linearly separable problems like XOR.

% Define XOR inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 0];

% Create a feedforward network with one hidden layer (perceptron doesn't support hidden layers directly)
net = feedforwardnet(2); % 2 neurons in the hidden layer

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display output
disp('Outputs for XOR problem with hidden layer:');
disp(output);
