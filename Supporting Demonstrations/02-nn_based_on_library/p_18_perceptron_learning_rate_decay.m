% p_18_perceptron_learning_rate_decay.m
% Model Architecture: Single-layer perceptron with 2 input neurons and 1 output neuron.
% Task: Introduce learning rate decay during training.
% Purpose: Demonstrates the effect of learning rate decay on training performance.

% Define inputs and targets
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron
net = perceptron;

% Set learning rate decay
net.trainParam.lr_dec = 0.9; % Decrease learning rate by 10% after each epoch

% Train the network
net = train(net, inputs, targets);

% Test the network
output = net(inputs);

% Display output
disp('Outputs with learning rate decay:');
disp(output);
