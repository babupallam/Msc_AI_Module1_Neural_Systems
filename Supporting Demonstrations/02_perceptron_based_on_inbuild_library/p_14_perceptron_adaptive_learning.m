% p_14_perceptron_adaptive_learning.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Demonstrate the use of an adaptive learning rate.
% Purpose: Shows how adaptive learning rate (increasing and decreasing) influences the training of a perceptron.

% Define inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron with adaptive learning rate
net = perceptron;
net.trainParam.lr_inc = 1.05; % Learning rate increment
net.trainParam.lr_dec = 0.7;  % Learning rate decrement

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display output
disp('Outputs with adaptive learning rate:');
disp(output);
