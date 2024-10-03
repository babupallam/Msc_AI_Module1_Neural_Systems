% File: perceptron_main_dynamic.m
% Purpose: Train and dynamically visualize perceptron decision boundary

clc;
clear;

% Load the Iris dataset
load fisheriris;

% Use only two classes (Setosa and Versicolor) and two features for simplicity
X = meas(1:100, 1:2);
species = species(1:100);

% Convert class labels to binary (-1 and 1)
y = strcmp(species, 'setosa'); % Setosa is 1, Versicolor is 0
y = 2*y - 1; % Convert logical values to (-1, 1)

% Normalize the features for better performance
X = normalize(X);

% Define hyperparameters
learning_rate = 0.01;
epochs = 20;

% Train the perceptron with dynamic visualization
weights = perceptron_train_dynamic(X, y, learning_rate, epochs);

% Display final weights
disp('Final weights:');
disp(weights);
