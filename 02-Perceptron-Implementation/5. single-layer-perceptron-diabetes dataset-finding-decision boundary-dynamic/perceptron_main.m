% File: perceptron_main.m
% Purpose: Train and test the perceptron using the Pima Indians Diabetes dataset with enhanced visualization

clc;
clear;

% Load the Pima Indians Diabetes dataset
data = readtable('diabetes.csv'); % Load data from CSV file
X = table2array(data(:, 1:end-1)); % Extract features
y = table2array(data(:, end)); % Extract labels (0 = non-diabetic, 1 = diabetic)

% Convert labels from (0, 1) to (-1, 1) for perceptron
y(y == 0) = -1;

% Normalize the features (important for gradient-based algorithms)
X = normalize(X);

% Define hyperparameters
learning_rate = 0.01;
epochs = 50;

% Initialize variables for tracking accuracy over epochs
accuracy_over_time = zeros(epochs, 1);

% Train the perceptron on the diabetes dataset with accuracy tracking
weights = zeros(size(X, 2) + 1, 1); % Initialize weights (including bias)

for epoch = 1:epochs
    weights = perceptron_train(X, y, learning_rate, 1); % Train for one epoch
    predictions = perceptron_predict(X, weights); % Make predictions
    accuracy_over_time(epoch) = sum(predictions == y) / length(y) * 100; % Store accuracy
end

% Test the perceptron on the same dataset
final_predictions = perceptron_predict(X, weights);

% Display the final weights
disp('Final weights:');
disp(weights);

% Calculate final accuracy
final_accuracy = sum(final_predictions == y) / length(y) * 100;
fprintf('Final Accuracy: %.2f%%\n', final_accuracy);

% Confusion matrix visualization
figure;
confusionchart(y, final_predictions);
title('Confusion Matrix for Perceptron on Pima Indians Diabetes Dataset');

% Plot accuracy over epochs
figure;
plot(1:epochs, accuracy_over_time, 'b-', 'LineWidth', 2);
title('Perceptron Accuracy Over Epochs');
xlabel('Epoch');
ylabel('Accuracy (%)');
grid on;

% Feature Correlation Matrix
figure;
corr_matrix = corr(X);
imagesc(corr_matrix); % Display correlation matrix as heatmap
colorbar;
title('Feature Correlation Matrix');
xticks(1:size(X,2)); yticks(1:size(X,2));
xticklabels(data.Properties.VariableNames(1:end-1)); % Use feature names as labels
yticklabels(data.Properties.VariableNames(1:end-1));

% Plot decision boundary using only two features (for visualization)
figure;
hold on;
X_two_features = X(:, 1:2); % Use first two features for simplicity
weights_two_features = perceptron_train(X_two_features, y, learning_rate, epochs);
final_predictions_two_features = perceptron_predict(X_two_features, weights_two_features);

% Plot class samples
scatter(X_two_features(y == 1, 1), X_two_features(y == 1, 2), 'bo', 'filled');
scatter(X_two_features(y == -1, 1), X_two_features(y == -1, 2), 'ro', 'filled');

% Plot decision boundary
x_vals = min(X_two_features(:, 1)):0.01:max(X_two_features(:, 1));
y_vals = -(weights_two_features(1) * x_vals + weights_two_features(3)) / weights_two_features(2);
plot(x_vals, y_vals, 'k-', 'LineWidth', 2);

title('Perceptron Decision Boundary with First Two Features');
xlabel('Feature 1 (Normalized)');
ylabel('Feature 2 (Normalized)');
legend('Diabetic', 'Non-Diabetic', 'Decision Boundary');
hold off;
