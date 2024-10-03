% TEST_SINGLE_LAYER_NN: Script to test the functionality of a single-layer 
% neural network. It uses multiple neurons with predefined weights and biases 
% to compute the output of the entire layer.

% Define the input vector (same as input for all neurons in the layer)
inputs = [0.3, 0.6, -0.1];  % Example inputs for the network

% Define the weights matrix where each row corresponds to a neuron's weights
weights = [0.2, -0.3, 0.4;   % Weights for neuron 1
          -0.5, 0.1, 0.2];   % Weights for neuron 2

% Define the bias vector (one bias for each neuron)
bias = [0.1, -0.2];  % Bias for neuron 1 and neuron 2

% Call the single_layer_nn function to compute the layer's output
output = single_layer_nn(inputs, weights, bias);

% Display the result to the user
disp('Single Layer NN Output:');
disp(output);  % Print the output as a vector
