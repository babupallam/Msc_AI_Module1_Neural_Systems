% p_22_perceptron_iris_classification.m
% Real-world perceptron implementation using the built-in Iris dataset in MATLAB
% Task: Classify iris flowers into different species using a simple perceptron

%% Section 1: Load the Iris dataset
% MATLAB has a built-in Iris dataset. It contains 150 samples of iris flowers, 
% with 4 features (sepal length, sepal width, petal length, petal width) and 3 target classes.

% Load the Iris dataset from CSV using readtable (handles text and numbers)
data = readtable('iris.csv', 'ReadVariableNames', false); % Don't read variable names if not present

% Extract input features (columns 1-4) and convert them to a numeric matrix
inputs = data{:, 1:4}';  % Extract first 4 columns and transpose to match input format

% Extract species labels (5th column)
species = data{:, 5};  % Extract species names (text)

% Convert species labels from text to numeric values (1 = setosa, 2 = versicolor, 3 = virginica)
species = categorical(species);  % Convert species names to categorical
targets = grp2idx(species);      % Convert categorical data to numeric indices (1, 2, 3)

% Display the first few rows of the input features and corresponding targets
disp('First few input features:');
disp(inputs(:, 1:5));

disp('First few target labels (numeric):');
disp(targets(1:5));


%% Section 2: Preprocess the data
% We reduce the problem to binary classification by focusing on one class (e.g., 'setosa') 
% and classify it against the rest ('versicolor' and 'virginica').
% Here, we'll classify 'setosa' (class 1) versus non-setosa (class 2 and 3).

% Convert the multi-class problem into a binary classification:
binaryTargets = (targets == 1); % 1 if 'setosa', 0 if 'versicolor' or 'virginica'

% Normalize the input features
inputs = normalize(inputs); % Normalize the features to improve training performance

% Display the first few rows of normalized data
disp('First few normalized feature vectors:');
disp(inputs(:, 1:5));

%% Section 3: Split the data into training and testing sets
% We'll split the dataset into 70% training and 30% testing.

[trainInd, ~, testInd] = dividerand(size(inputs, 2), 0.7, 0, 0.3); % Random split
trainInputs = inputs(:, trainInd);   % Training input features
trainTargets = binaryTargets(trainInd);  % Training labels
trainTargets = trainTargets(:)';     % Ensure targets are a row vector
testInputs = inputs(:, testInd);     % Test input features
testTargets = binaryTargets(testInd);    % Test labels
testTargets = testTargets(:)';% Ensure targets are a row vector


% Display the size of the training and testing sets
disp(['Number of training examples: ', num2str(length(trainInd))]);
disp(['Number of testing examples: ', num2str(length(testInd))]);

% Debugging: Display sizes to check if inputs and targets have the same number of samples
disp(['Size of trainInputs: ', num2str(size(trainInputs, 1)), ' x ', num2str(size(trainInputs, 2))]);
disp(['Size of trainTargets: ', num2str(size(trainTargets, 1)), ' x ', num2str(size(trainTargets, 2))]);

%% Section 4: Create and train the perceptron
% Create a perceptron and train it using the training data.

net = perceptron;            % Create the perceptron
net = train(net, trainInputs, trainTargets); % Train the perceptron on training data

% Display that training is completed
disp('Perceptron training completed.');

%% Section 5: Test the perceptron on the test data
% Test the trained perceptron on the test data to evaluate its performance.

testOutputs = net(testInputs); % Get the perceptron's predictions on the test data

% Convert the perceptron's continuous outputs to binary (0 or 1)
testOutputs = round(testOutputs);

% Display the first few predicted outputs and actual test targets
disp('First few test outputs (predicted vs actual):');
disp([testOutputs(1:5); testTargets(1:5)]); % Show predicted vs actual labels

%% Section 6: Calculate accuracy
% Calculate the accuracy of the perceptron model by comparing predicted and actual values.

accuracy = sum(testOutputs == testTargets) / length(testTargets) * 100; % Accuracy calculation

% Display the accuracy
disp(['Perceptron accuracy on test data: ', num2str(accuracy), '%']);

%% Section 7: Visualize the confusion matrix
% A confusion matrix helps visualize how well the perceptron is classifying the data.

figure; % Create a new figure
plotconfusion(testTargets, testOutputs); % Plot the confusion matrix
title('Confusion Matrix for Iris Classification (Setosa vs Non-Setosa)');
