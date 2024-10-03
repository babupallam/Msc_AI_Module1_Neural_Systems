% TEST_NEURON_MODEL: Script to test the functionality of a single neuron.
% It provides some input data, weights, and bias, and uses the neuron_model
% function to calculate and display the output.

% Define the input vector (example features for the neuron)
input = [0.5, -0.2, 0.1];  % Example input values for the neuron

% Define the weight vector (same size as input)
weights = [0.4, -0.6, 0.1];  % Example weights assigned to the inputs

% Define the bias (scalar)
bias = 0.2;  % Example bias term

% Call the neuron_model function to compute the output
output = neuron_model(input, weights, bias);

% Display the result to the user
disp(['Neuron Output: ', num2str(output)]);
