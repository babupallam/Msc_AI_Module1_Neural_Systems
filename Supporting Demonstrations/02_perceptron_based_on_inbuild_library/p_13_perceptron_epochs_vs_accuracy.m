% p_13_perceptron_epochs_vs_accuracy.m
% Model Architecture: Single-layer perceptron one neuron with 2 input and 1 output.
% Task: Study the effect of varying the number of epochs (iterations) on training accuracy.
% Purpose: Demonstrates the relationship between training epochs and model accuracy.

% Define inputs and targets for OR gate
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 1];

% Create a perceptron
net = perceptron;

% Set maximum epochs for training
net.trainParam.epochs = 1; % Initially, train for 1 epoch

% Train and test the perceptron for increasing epochs
for i = 1:5
    net.trainParam.epochs = i;
    net = train(net, inputs, targets);
    
    % Test the network
    output = net(inputs);
    
    % Display output
    disp(['Outputs after ', num2str(i), ' epochs:']);
    disp(output);
end
