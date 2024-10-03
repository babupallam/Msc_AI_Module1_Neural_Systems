% File: perceptron_predict.m
% Purpose: Predict labels for new samples using the trained perceptron

function predictions = perceptron_predict(X, weights)
    % X: Input features matrix (each row is a sample, each column a feature)
    % weights: The learned weights from the perceptron
    % predictions: Predicted labels for the input samples
    
    [n_samples, n_features] = size(X);
    
    % Add bias term to the input matrix (last column all ones)
    X = [X, ones(n_samples, 1)];

    % Predict the output using the sign of the weighted sum
    predictions = sign(X * weights);
end
