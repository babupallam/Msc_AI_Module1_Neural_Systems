% p_15_perceptron_momentum.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Simulate the effect of momentum on training.
% Purpose: Demonstrates how adding momentum can accelerate the convergence of the perceptron.

% Define inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron
net = perceptron;

% Add momentum (simulated, as perceptron in MATLAB does not natively support momentum)
net.trainParam.mc = 0.9; % Momentum coefficient

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display output
disp('Outputs with momentum:');
disp(output);
