% p_11_perceptron_with_noise.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Classify noisy data (OR gate with added noise to inputs).
% Purpose: Demonstrates the robustness of the perceptron when trained on noisy data.

% Define inputs and targets for OR gate
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Add some noise to the inputs
noise = 0.1 * randn(size(inputs));
inputs_noisy = inputs + noise;

% Create a perceptron
net = perceptron;

% Train the perceptron with noisy data
net = train(net, inputs_noisy, targets);

% Test the network
output = net(inputs);

% Display output
disp('Outputs with noisy inputs:');
disp(output);

% Visualize the noisy inputs and decision boundary
figure;
plotpv(inputs_noisy, targets);
plotpc(net.IW{1}, net.b{1});
title('Perceptron with Noisy Data');
