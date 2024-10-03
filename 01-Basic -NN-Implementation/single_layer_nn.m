function output = single_layer_nn(inputs, weights, bias)
    % SINGLE_LAYER_NN: This function simulates forward propagation in a 
    % single-layer neural network (without hidden layers). 
    % Each neuron in the layer has its own weights and bias.
    %
    % Inputs:
    % inputs - A vector of input values
    % weights - A matrix of weights where each row corresponds to 
    %           the weights for a neuron in the layer
    % bias - A vector of bias values (one for each neuron)
    %
    % Output:
    % output - A vector of outputs for each neuron in the layer
    
    % Step 1: Determine the number of neurons in the layer
    n_neurons = size(weights, 1);  % Number of rows = number of neurons
    
    % Step 2: Preallocate a vector to store the output of each neuron
    output = zeros(1, n_neurons);
    
    % Step 3: Loop over each neuron and compute its output
    for i = 1:n_neurons
        % For each neuron, call the neuron_model function with its weights and bias
        output(i) = neuron_model(inputs, weights(i, :), bias(i));
    end
end
