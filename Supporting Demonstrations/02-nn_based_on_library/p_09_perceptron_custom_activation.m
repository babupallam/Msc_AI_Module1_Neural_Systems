% p_09_perceptron_custom_activation.m
% Model Architecture: 
% A single-layer perceptron with 2 input neurons (for 2D points) and 1 output neuron.
% The activation function used in this perceptron is customized to be a linear function 
% ('purelin') instead of the default hard-limit function ('hardlim').

% Task:
% We will modify the activation function of the perceptron and observe the 
% behavior of the output with the new activation function. 
% The task is to train the perceptron to approximate an OR gate, but with 
% a linear output function.

% Purpose:
% This code demonstrates how to change the activation function of a perceptron. 
% Normally, perceptrons use a step function or hard limit function for binary classification.
% Here, we will use a linear activation function ('purelin') to observe the behavior 
% of the output, which will no longer be binary.

% Define inputs (4 points in a 2D space)
% Each column represents an input vector, which is a pair of (x1, x2)
% Inputs represent the combinations for a logical OR gate (00, 01, 10, 11).
inputs = [0 0;   % Input 1: [0, 0]
          0 1;   % Input 2: [0, 1]
          1 0;   % Input 3: [1, 0]
          1 1]'; % Input 4: [1, 1]
      
% Define targets (the correct output for each input)
% These are the target outputs for an OR gate. 
% For OR logic: 
% - 0 OR 0 = 0
% - 0 OR 1 = 1
% - 1 OR 0 = 1
% - 1 OR 1 = 1
targets = [0 1 1 1];

% Create a perceptron
% By default, MATLAB's perceptron uses a hard-limit activation function ('hardlim').
net = perceptron;

% Display the default activation function for the output layer
disp('Default activation function:');
disp(net.layers{1}.transferFcn);

% Set the custom activation function
% We are changing the transfer (activation) function from the default 'hardlim' to 'purelin' (linear).
net.layers{1}.transferFcn = 'purelin'; % 'purelin' makes the output a linear combination of inputs

% Display the updated activation function
disp('Updated activation function:');
disp(net.layers{1}.transferFcn);

% Train the network
% We will train the perceptron using the inputs and corresponding targets.
% The network will adjust the weights and bias during training to try and match the target outputs.
net = train(net, inputs, targets);

% Test the network after training
% Once trained, we pass the same inputs to the network to see how it performs.
output = net(inputs);

% Display the output of the perceptron after training
% Since we are using a linear activation function, the outputs will not be limited to 0 or 1.
% We expect real numbers (linear outputs), unlike the binary output of a traditional perceptron.
disp('Outputs with custom linear activation (purelin):');
disp(output);

% Visualize the trained perceptron
% This will show how the perceptron has learned to separate the input space.
% The decision boundary will be plotted along with the input points.
figure;
plotpv(inputs, targets);   % Plot input vectors and target values
plotpc(net.IW{1}, net.b{1}); % Plot the perceptron decision boundary based on its weights
title('Perceptron with Linear Activation Function (purelin)');

% Key explanation:
% By changing the activation function from 'hardlim' (which produces binary output)
% to 'purelin' (which produces linear output), we observe that the output of the perceptron
% is no longer limited to binary values (0 or 1). Instead, it produces continuous values
% that are a linear combination of the inputs and the learned weights.
