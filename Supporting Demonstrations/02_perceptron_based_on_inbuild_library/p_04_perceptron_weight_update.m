% p_04_perceptron_weight_update.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Demonstrate weight and bias updates during training.
% Purpose: Shows how perceptron weights and bias change as the model is trained.

% Define inputs and targets for OR gate
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron
net = perceptron;

% Visualize initial weights and bias
disp('Initial weights and bias:');
disp(net.IW{1});
disp(net.b{1});

% Train the network
net = train(net, inputs, targets);

% Visualize updated weights and bias
disp('Updated weights and bias after training:');
disp(net.IW{1});
disp(net.b{1});

% Test the network
output = net(inputs);

% Display output
disp('Outputs after training:');
disp(output);
