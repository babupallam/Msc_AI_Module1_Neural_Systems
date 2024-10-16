% ============================================================
% Perceptron for Multi-Class Classification (One-vs-All Approach)
% ============================================================
% Description:
% This code implements a multi-class classification perceptron using a
% one-vs-all approach in MATLAB. The perceptron is trained to classify
% inputs into one of three classes.
%
% Key Characteristics:
% 1. Type: Multi-Class Perceptron (One-vs-All)
%    - A basic perceptron that performs binary classification, but uses
%      multiple binary classifiers (one for each class) to achieve multi-class classification.
%
% 2. Layers: 1 Layer
%    - The network consists of a single output layer with one binary classifier
%      for each class.
%
% 3. Neurons: Multiple Binary Classifiers
%    - Each classifier is a perceptron that learns to distinguish its class from others.
%
% Problem: 
% This perceptron is trained to classify inputs into three different classes.
% It uses the one-vs-all approach where each class has its own perceptron model.
% ============================================================

% Define the input data
% Four inputs, each associated with one of three classes
inputs = [1 0; 0 1; 0 0; 1 1];  % Each row is an input vector (2D)

% Define the target outputs (multi-class labels)
% Three classes represented as 1, 2, and 3
targets = [1; 2; 3; 1];  % Expected class for each input

% Number of classes
num_classes = 3;

% Number of features (input dimensions)
num_features = size(inputs, 2);

% Initialize the weights for each class (one perceptron for each class)
% Weights matrix: Each row represents the weights for one classifier
weights = rand(num_classes, num_features);

% Initialize the bias for each class
bias = rand(num_classes, 1);

% Set the learning rate
learning_rate = 0.1;

% Set the number of epochs (training iterations)
epochs = 10;

% ============================================================
% Multi-Class Perceptron Training (One-vs-All)
% ============================================================

% Begin training loop for the given number of epochs
for epoch = 1:epochs
    fprintf('\nEpoch %d\n', epoch);
    
    % For each input
    for i = 1:size(inputs, 1)%4
        % For each class, compute the net input and make a prediction
        net_inputs = zeros(num_classes, 1);  % Initialize the net inputs for each class
        outputs = zeros(num_classes, 1);     % Initialize the predicted outputs for each class
        
        % Compute net input and output for each classifier (one for each class)
        for class = 1:num_classes
            net_inputs(class) = dot(weights(class, :), inputs(i, :)) + bias(class);
            outputs(class) = net_inputs(class) > 0;  % Step function (binary prediction)
        end
        
        % Determine which class has the highest score (predicted class)
        [~, predicted_class] = max(net_inputs);
        
        % One-vs-all error correction: Update only the classifier for the correct class
        for class = 1:num_classes
            % Target is 1 if this is the correct class, 0 otherwise
            target = (targets(i) == class);
            
            % Calculate the error for this class
            error = target - outputs(class);
            
            % Update weights and bias for this classifier based on the error
            weights(class, :) = weights(class, :) + learning_rate * error * inputs(i, :);
            bias(class) = bias(class) + learning_rate * error;
            
            % Display updates for the current class
            fprintf('Input: [%d %d], Target Class: %d, Predicted Class: %d, Classifier: %d, Error: %d\n', ...
                    inputs(i, 1), inputs(i, 2), targets(i), predicted_class, class, error);
            fprintf('Updated Weights for Class %d: [%.4f, %.4f], Updated Bias: %.4f\n', class, weights(class, 1), weights(class, 2), bias(class));
        end
    end
end

% ============================================================
% Model Testing Phase
% ============================================================

% Define the testing data (same as training data in this case)
test_inputs = [1 0; 0 1; 0 0; 1 1];  % Input patterns to test
test_targets = [1; 2; 3; 1];  % Expected class labels for each input

% Test the perceptron on each input in the test set
fprintf('\nTesting the trained multi-class perceptron:\n');
for i = 1:size(test_inputs, 1)
    % Compute net input for each class and determine the predicted class
    net_inputs = zeros(num_classes, 1);
    
    for class = 1:num_classes
        net_inputs(class) = dot(weights(class, :), test_inputs(i, :)) + bias(class);
    end
    
    % The predicted class is the one with the highest net input
    [~, predicted_class] = max(net_inputs);
    
    % Display the test input, predicted class, and expected class
    fprintf('Test Input: [%d %d], Predicted Class: %d, Expected Class: %d\n', ...
            test_inputs(i, 1), test_inputs(i, 2), predicted_class, test_targets(i));
end
