% File: perceptron_main.m
% Purpose: Main script to train and test the perceptron with example data

clc;
clear;

% Example dataset (linearly separable)
% Two classes: Class 1 (1, 1) and (2, 2), Class -1 (-1, -1) and (-2, -2)
X = [1 1; 2 2; -1 -1; -2 -2];
y = [1; 1; -1; -1]; % Labels: 1 for class 1, -1 for class -1

% Define hyperparameters
learning_rate = 0.1;
epochs = 10;

% Train the perceptron
weights = perceptron_train(X, y, learning_rate, epochs);

% Test the perceptron on the same dataset (for simplicity)
predictions = perceptron_predict(X, weights);

% Display the results
disp('Final weights:');
disp(weights);

disp('Predictions:');
disp(predictions);

% Plot the results and decision boundary
figure;
hold on;

% Plot samples from both classes
scatter(X(y == 1, 1), X(y == 1, 2), 'bo', 'filled');
scatter(X(y == -1, 1), X(y == -1, 2), 'ro', 'filled');

% Plot decision boundary
x_vals = -3:0.1:3;
y_vals = -(weights(1) * x_vals + weights(3)) / weights(2);
plot(x_vals, y_vals, 'k-');

title('Perceptron Decision Boundary');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class -1', 'Decision Boundary');
hold off;
