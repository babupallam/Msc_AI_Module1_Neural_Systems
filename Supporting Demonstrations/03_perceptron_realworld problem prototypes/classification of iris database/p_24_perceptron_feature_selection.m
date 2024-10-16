% p_24_perceptron_feature_selection.m
% Implement feature selection for the perceptron
% Task: Classify iris flowers into different species using a perceptron after selecting relevant features

%% Section 1: Load the Iris dataset
% Load the Iris dataset from CSV using readtable (handles text and numbers)
data = readtable('iris.csv', 'ReadVariableNames', false); % Don't read variable names if not present

% Extract input features (columns 1-4) and convert them to a numeric matrix
inputs = data{:, 1:4}';  % Extract first 4 columns and transpose to match input format

% Extract species labels (5th column)
species = data{:, 5};  % Extract species names (text)

% Convert species labels from text to numeric values (1 = setosa, 2 = versicolor, 3 = virginica)
species = categorical(species);  % Convert species names to categorical
targets = grp2idx(species);      % Convert categorical data to numeric indices (1, 2, 3)

% Convert the multi-class problem into a binary classification (Setosa vs Non-Setosa)
binaryTargets = (targets == 1); % 1 if 'setosa', 0 if 'versicolor' or 'virginica'

%% Section 2: Feature Selection
% We will use only two features: Petal Length (3rd column) and Petal Width (4th column)
% These features are known to be more effective for distinguishing between classes
selectedInputs = inputs([3, 4], :); % Select petal length and petal width as features

% Display the first few rows of selected features
disp('First few selected feature vectors (Petal Length, Petal Width):');
disp(selectedInputs(:, 1:5));

%% Section 3: Normalize Selected Features
% Normalize the selected features for better training performance
selectedInputs = normalize(selectedInputs); % Normalize the selected features

% Display the first few normalized feature vectors
disp('First few normalized selected feature vectors:');
disp(selectedInputs(:, 1:5));

%% Section 4: Split the data into training and testing sets
% Split the dataset into 70% training and 30% testing.
[trainInd, ~, testInd] = dividerand(size(selectedInputs, 2), 0.7, 0, 0.3); % Random split
trainInputs = selectedInputs(:, trainInd);   % Training input features
trainTargets = binaryTargets(trainInd);  % Training labels
trainTargets = trainTargets(:)';     % Ensure targets are a row vector
testInputs = selectedInputs(:, testInd);     % Test input features
testTargets = binaryTargets(testInd);    % Test labels
testTargets = testTargets(:)'; % Ensure targets are a row vector

%% Section 5: Create and Train the Perceptron
% Initialize and train the perceptron with the selected features
net = perceptron;  % Create perceptron
net = train(net, trainInputs, trainTargets); % Train the perceptron on training data

% Display that training is completed
disp('Perceptron training completed with selected features.');

%% Section 6: Test the Perceptron on the Test Data
% Test the trained perceptron on the test data
testOutputs = net(testInputs); % Get the perceptron's predictions on the test data
testOutputs = round(testOutputs); % Convert continuous outputs to binary (0 or 1)

% Display the first few predicted outputs and actual test targets
disp('First few test outputs (predicted vs actual):');
disp([testOutputs(1:5); testTargets(1:5)]); % Show predicted vs actual labels

%% Section 7: Calculate Accuracy
% Calculate the accuracy of the perceptron model using the selected features
accuracy = sum(testOutputs == testTargets) / length(testTargets) * 100; % Accuracy calculation

% Display the accuracy
disp(['Perceptron accuracy with selected features (Petal Length and Petal Width): ', num2str(accuracy), '%']);

%% Section 8: Visualize Decision Boundary (Optional)
% Visualization of the decision boundary using the selected two features
% This helps to better understand how the perceptron separates the classes

% Generate a grid for plotting
[x1Grid, x2Grid] = meshgrid(linspace(min(selectedInputs(1,:)), max(selectedInputs(1,:)), 100), ...
                            linspace(min(selectedInputs(2,:)), max(selectedInputs(2,:)), 100));
gridPoints = [x1Grid(:)'; x2Grid(:)'];

% Test perceptron on the grid points
gridOutputs = net(gridPoints);
gridOutputs = round(gridOutputs);

% Plot the decision boundary and data points
figure;
gscatter(selectedInputs(1,:), selectedInputs(2,:), binaryTargets, 'rb', 'xo');
hold on;
contour(x1Grid, x2Grid, reshape(gridOutputs, size(x1Grid)), [0.5, 0.5], 'k');
title('Perceptron Decision Boundary (Petal Length vs Petal Width)');
xlabel('Petal Length');
ylabel('Petal Width');
