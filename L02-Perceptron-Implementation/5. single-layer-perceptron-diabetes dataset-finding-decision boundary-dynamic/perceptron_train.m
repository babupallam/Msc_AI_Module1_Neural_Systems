% File: perceptron_train.m
% Purpose: Train a single-layer perceptron using linearly separable data

function weights = perceptron_train(X, y, learning_rate, epochs)
    % X: Input features matrix (each row is a sample, each column a feature)
    % y: Labels (1 or -1 for each sample)
    % learning_rate: The step size for weight updates
    % epochs: Number of times to iterate over the entire dataset
    % weights: The learned weights after training

    [n_samples, n_features] = size(X);
    
    % Initialize weights (including bias as the last weight)
    weights = zeros(n_features + 1, 1);
    
    % Add bias term to the input matrix (last column all ones)
    X = [X, ones(n_samples, 1)];

    % Training loop
    for epoch = 1:epochs
        for i = 1:n_samples
            % Predict output using current weights
            prediction = sign(X(i, :) * weights);
            
            % Update weights if the prediction is incorrect
            if prediction ~= y(i)
                % Update rule: weights = weights + learning_rate * error * input
                weights = weights + learning_rate * y(i) * X(i, :)';
            end
        end
        fprintf('Epoch %d completed.\n', epoch);
    end
end
