% Purpose: Train and test the perceptron using the Diabetes dataset
% and dynamically visualize the decision boundary.

clc;
clear;

% Load the Pima Indians Diabetes dataset
data = readtable('diabetes.csv'); % Load data from CSV file
X = table2array(data(:, 1:2)); % Extract first two features for visualization
y = table2array(data(:, end)); % Extract labels (0 = non-diabetic, 1 = diabetic)

% Convert labels from (0, 1) to (-1, 1) for perceptron
y(y == 0) = -1;

% Normalize the features (important for gradient-based algorithms)
X = normalize(X);

% Define hyperparameters
learning_rate = 0.01;
epochs = 50;

% Initialize weights (including bias)
[n_samples, n_features] = size(X);
weights = zeros(n_features + 1, 1);

% Add bias term to the input matrix (column of ones)
X = [X, ones(n_samples, 1)];

% Prepare figure for dynamic decision boundary visualization
figure;
hold on;

% Scatter plot of the two classes
scatter(X(y == 1, 1), X(y == 1, 2), 'bo', 'filled'); % Diabetic
scatter(X(y == -1, 1), X(y == -1, 2), 'ro', 'filled'); % Non-Diabetic
x_vals = linspace(min(X(:,1)) - 1, max(X(:,1)) + 1, 100); % For plotting decision boundary

title('Dynamic Perceptron Decision Boundary');
xlabel('Feature 1 (Normalized)');
ylabel('Feature 2 (Normalized)');
legend('Diabetic', 'Non-Diabetic', 'Decision Boundary');

% Training loop with dynamic visualization
for epoch = 1:epochs
    % Loop through all samples in the dataset
    for i = 1:n_samples
        % Predict output using the current weights
        prediction = sign(X(i, :) * weights);
        
        % Update weights if the prediction is incorrect
        if prediction ~= y(i)
            % Update rule: weights = weights + learning_rate * error * input
            weights = weights + learning_rate * y(i) * X(i, :)';
        end
    end
    
    % Plot the decision boundary dynamically after each epoch
    % Decision boundary equation: w1*x1 + w2*x2 + w3 = 0 -> x2 = -(w1*x1 + w3) / w2
    y_vals = -(weights(1) * x_vals + weights(3)) / weights(2);
    
    % Clear previous decision boundary line before plotting a new one
    if epoch > 1
        delete(h);
    end
    
    % Plot the new decision boundary
    h = plot(x_vals, y_vals, 'k-', 'LineWidth', 2);
    
    % Update title to show the current epoch
    title(sprintf('Epoch %d - Dynamic Perceptron Decision Boundary', epoch));
    
    % Pause for a short time to visually see the update
    pause(0.1);
    
    % Check if the perceptron has perfectly classified the data
    predictions = sign(X * weights);
    if all(predictions == y)
        disp('Converged! Separating hyperplane found.');
        break;
    end
end

hold off;

% Final predictions and performance evaluation
final_predictions = sign(X * weights);
accuracy = sum(final_predictions == y) / n_samples * 100;
fprintf('Final Accuracy: %.2f%%\n', accuracy);

% Plot the confusion matrix
figure;
confusionchart(y, final_predictions);
title('Confusion Matrix for Perceptron on Pima Indians Diabetes Dataset');
