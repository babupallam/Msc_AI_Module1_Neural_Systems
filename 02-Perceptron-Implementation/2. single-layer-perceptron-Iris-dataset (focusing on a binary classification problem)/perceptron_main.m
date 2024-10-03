% File: perceptron_main.m
% Purpose: Train and test the perceptron using a subset of the Iris dataset

clc;
clear;

% Load the Iris dataset
load fisheriris; % This loads 'meas' (features) and 'species' (labels)

% We will use only the first two classes for binary classification
% Classes: "setosa" and "versicolor" (ignore "virginica" class)
X = meas(1:100, 1:2); % Take only the first two features for simplicity
species = species(1:100);

% Convert class labels to binary (-1 and 1)
y = strcmp(species, 'setosa'); % Setosa is 1, Versicolor is 0
y = 2*y - 1; % Convert logical values (0,1) to (-1,1)

% Normalize the features (optional but recommended for better training)
X = normalize(X);

% Define hyperparameters
learning_rate = 0.01;
epochs = 20;

% Train the perceptron on the extended dataset
weights = perceptron_train(X, y, learning_rate, epochs);

% Test the perceptron on the same dataset
predictions = perceptron_predict(X, weights);

% Display the final weights and predictions
disp('Final weights:');
disp(weights);

disp('Predictions:');
disp(predictions);

% Calculate and display accuracy
accuracy = sum(predictions == y) / length(y) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);

% Plot the results and decision boundary
figure;
hold on;

% Plot samples from both classes
scatter(X(y == 1, 1), X(y == 1, 2), 'bo', 'filled'); % Setosa
scatter(X(y == -1, 1), X(y == -1, 2), 'ro', 'filled'); % Versicolor

% Plot decision boundary
x_vals = min(X(:,1)):0.01:max(X(:,1));
y_vals = -(weights(1) * x_vals + weights(3)) / weights(2); % decision boundary formula
plot(x_vals, y_vals, 'k-', 'LineWidth', 2);

title('Perceptron Decision Boundary (Iris Dataset)');
xlabel('Sepal Length (Normalized)');
ylabel('Sepal Width (Normalized)');
legend('Setosa', 'Versicolor', 'Decision Boundary');
hold off;
