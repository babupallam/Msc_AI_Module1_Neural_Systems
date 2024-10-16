% p_05_perceptron_learning_rate.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Investigate the effect of different learning rates.
% Purpose: Demonstrates how learning rate influences perceptron training.

% Define inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron with a custom learning rate
net = perceptron;
net.trainParam.lr = 0.1; % Set learning rate to 0.1

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display output
disp('Outputs after training with learning rate 0.1:');
disp(output);

% Visualize the network performance
figure;
plotpv(inputs, targets);
plotpc(net.IW{1}, net.b{1});
title('Perceptron with Learning Rate 0.1');
