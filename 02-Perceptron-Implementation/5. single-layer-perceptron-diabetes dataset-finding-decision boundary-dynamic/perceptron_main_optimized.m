% File: perceptron_main_optimized.m
% Purpose: Train and test an optimized perceptron on the Pima Indians Diabetes dataset
% Steps to Optimize Perceptron:

% 1. Feature Selection:
%    We'll select the most relevant features that are highly correlated 
%    with the target label. This reduces noise and focuses on the most 
%    informative features for better learning.

% 2. Polynomial Features:
%    Introduce higher-order terms, such as squares and interactions of 
%    features, to allow the perceptron to fit more complex patterns. 
%    This helps the perceptron deal with non-linearly separable data by 
%    creating quadratic decision boundaries.

% 3. Tuning Learning Rate and Epochs:
%    We’ll experiment with different learning rates to ensure the perceptron 
%    converges properly without overshooting. Additionally, increasing the 
%    number of epochs allows the perceptron to fully converge over more training 
%    iterations.


clc;
clear;

% Load the Pima Indians Diabetes dataset
data = readtable('diabetes.csv'); % Load data from CSV file
X = table2array(data(:, 1:end-1)); % Extract features
y = table2array(data(:, end)); % Extract labels (0 = non-diabetic, 1 = diabetic)

% Convert labels from (0, 1) to (-1, 1) for perceptron
y(y == 0) = -1;

% Step 1: Feature Selection (optional, can be manually selected)
% Select only the most correlated features based on correlation with label
% For demonstration, we manually select a subset of important features
% Feature indices 1 (Pregnancies), 2 (Glucose), and 6 (BMI) are commonly relevant
X_selected = X(:, [2, 6, 8]); % Selecting Glucose, BMI, Age

% Step 2: Polynomial Feature Expansion (to capture nonlinearity)
X_poly = [X_selected, X_selected.^2]; % Add squared terms of each selected feature

% Normalize the expanded features
X_poly = normalize(X_poly);

% Step 3: Define hyperparameters
learning_rate = 0.001; % Optimized learning rate
epochs = 100; % Increased epochs to allow convergence

% Initialize variables for tracking accuracy over epochs
accuracy_over_time = zeros(epochs, 1);

% Train the perceptron on the diabetes dataset with optimized settings
weights = zeros(size(X_poly, 2) + 1, 1); % Initialize weights (including bias)

for epoch = 1:epochs
    weights = perceptron_train(X_poly, y, learning_rate, 1); % Train for one epoch
    predictions = perceptron_predict(X_poly, weights); % Make predictions
    accuracy_over_time(epoch) = sum(predictions == y) / length(y) * 100; % Store accuracy
end

% Test the perceptron on the same dataset
final_predictions = perceptron_predict(X_poly, weights);

% Display the final weights
disp('Final weights:');
disp(weights);

% Calculate final accuracy
final_accuracy = sum(final_predictions == y) / length(y) * 100;
fprintf('Final Accuracy: %.2f%%\n', final_accuracy);

% Plot accuracy over epochs
figure;
plot(1:epochs, accuracy_over_time, 'b-', 'LineWidth', 2);
title('Perceptron Accuracy Over Epochs (Optimized)');
xlabel('Epoch');
ylabel('Accuracy (%)');
grid on;

% Confusion matrix visualization
figure;
confusionchart(y, final_predictions);
title('Confusion Matrix for Optimized Perceptron on Pima Indians Diabetes Dataset');
