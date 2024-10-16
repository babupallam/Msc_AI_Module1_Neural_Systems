% p_10_perceptron_batch_training.m
% Model Architecture: 
% - Single-layer perceptron with 2 input neurons (for 2D points) and 1 output neuron.
% - This perceptron is trained in batch mode, where the entire dataset is used to update the model weights in each epoch.
% 
% Task:
% - Train the perceptron to learn the OR gate logic using batch training.
% 
% Purpose:
% - This code demonstrates batch training in perceptrons, where the entire training set is used in each iteration, 
%   as opposed to incremental (online) training, where weights are updated after each input.
% 
% Key Takeaway:
% - Batch training is often faster for small datasets and helps the model converge more smoothly.

% Define inputs (each column represents an input vector)
% Inputs represent the combinations for a logical OR gate (00, 01, 10, 11).
inputs = [0 0;   % Input 1: [0, 0]
          0 1;   % Input 2: [0, 1]
          1 0;   % Input 3: [1, 0]
          1 1]'; % Input 4: [1, 1]

% Define targets (the correct output for each input)
% These are the target outputs for an OR gate. 
% OR logic:
% - 0 OR 0 = 0
% - 0 OR 1 = 1
% - 1 OR 0 = 1
% - 1 OR 1 = 1
targets = [0 1 1 1]; 

% Create a perceptron
% The default perceptron in MATLAB has a single layer with one output neuron.
net = perceptron;

% Set training to batch mode
% Batch mode means the perceptron will use the entire dataset in each iteration of training.
% This setting disables data division (meaning no separation into training/validation/testing subsets).
net.divideFcn = 'dividetrain'; % Use all the data for training (no data held out for validation or testing)

% Train the network
% The train function will now use batch training, processing all input vectors in each iteration.
% It will adjust the weights and biases based on the entire set of input-target pairs.
net = train(net, inputs, targets);

% Test the network after training
% After training is complete, we pass the same inputs through the network to see the predicted outputs.
output = net(inputs);

% Display the output of the perceptron after batch training
% Since this is an OR gate, we expect the perceptron to output values close to the target [0 1 1 1].
disp('Outputs after batch training:');
disp(output);

% Visualize the trained perceptron
% The plot will show the input vectors as points, the target values as desired output, and the decision boundary learned by the perceptron.
figure;
plotpv(inputs, targets);   % Plot input vectors and their target values
plotpc(net.IW{1}, net.b{1}); % Plot the decision boundary learned by the perceptron
title('Perceptron with Batch Training for OR Logic Gate');

% Key Explanation:
% - Batch training means that the perceptron adjusts its weights only after seeing the entire dataset.
% - This can help with convergence and avoids the noisy updates that occur with incremental (one-by-one) training.
% - The decision boundary learned by the perceptron is a straight line, which separates the input space into two regions:
%   one for inputs that produce a '0' output and one for inputs that produce a '1' output.
