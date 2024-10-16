% p_21_perceptron_diabetes_classification.m
% Real-world perceptron implementation using the Pima Indians Diabetes dataset
% Task: Classify patients as diabetic or non-diabetic using a simple perceptron

%% Section 1: Load the dataset
% The Pima Indians Diabetes dataset is stored in a CSV file where:
% Columns 1-8 are medical features, and column 9 is the binary target (1 = diabetes, 0 = non-diabetes)

data = readmatrix('pima-indians-diabetes.csv'); % Load the data from CSV

% Separate the inputs (features) and targets (class labels)
inputs = data(:, 1:8)'; % Medical features (glucose, BMI, age, etc.)
targets = data(:, 9)';  % Binary target (1 = diabetic, 0 = non-diabetic)

% Display the size of the data
disp(['Number of data points: ', num2str(size(inputs, 2))]);

%% Section 2: Normalize the input features
% Normalizing the input features ensures that they are on a similar scale, 
% which can improve the performance of the perceptron during training.

inputs = normalize(inputs); % Normalize each feature (column-wise)

% Display the first few rows of normalized data to check
disp('First few normalized inputs:');
disp(inputs(:, 1:5)); % Show first 5 normalized input examples

%% Section 3: Split the data into training and testing sets
% We'll split the dataset into 70% for training and 30% for testing. This helps
% evaluate how well the perceptron generalizes to unseen data.

[trainInd, ~, testInd] = dividerand(size(inputs, 2), 0.7, 0, 0.3); % Randomly divide data
trainInputs = inputs(:, trainInd);   % Training inputs
trainTargets = targets(trainInd);    % Training targets
testInputs = inputs(:, testInd);     % Testing inputs
testTargets = targets(testInd);      % Testing targets

% Display the size of the training and testing sets
disp(['Number of training examples: ', num2str(length(trainInd))]);
disp(['Number of testing examples: ', num2str(length(testInd))]);

%% Section 4: Create and train the perceptron
% We use MATLAB's perceptron function to create the model.
% The perceptron is trained using the training inputs and targets.

net = perceptron;            % Create the perceptron
net = train(net, trainInputs, trainTargets); % Train the perceptron on training data

% Display the training completion message
disp('Perceptron training completed.');

%% Section 5: Test the perceptron on the test data
% After training, we evaluate the perceptron on the test data to see how well it performs.

testOutputs = net(testInputs); % Get the network's output for the test data

% Convert the outputs to binary (0 or 1) to compare with actual targets
testOutputs = round(testOutputs);

% Display the predicted outputs and actual targets for the first few test examples
disp('First few test outputs (predicted vs actual):');
disp([testOutputs(1:5); testTargets(1:5)]); % Show predicted and actual values side by side

%% Section 6: Calculate accuracy
% Accuracy is the percentage of correct predictions out of the total number of test examples.

accuracy = sum(testOutputs == testTargets) / length(testTargets) * 100; % Calculate accuracy

% Display the calculated accuracy
disp(['Perceptron accuracy on test data: ', num2str(accuracy), '%']);

%% Section 7: Visualize the confusion matrix
% The confusion matrix helps visualize how well the perceptron is classifying the data.
% It shows the number of true positives, true negatives, false positives, and false negatives.

figure; % Create a new figure
plotconfusion(testTargets, testOutputs); % Plot the confusion matrix
title('Confusion Matrix for Diabetes Classification'); % Add a title

%% Section 8: Visualize the ROC Curve (Optional)
% The ROC curve is useful for understanding the trade-off between the true positive rate 
% and the false positive rate as the decision threshold changes.

% If the rocCurve function is available, we plot the ROC curve.
% The ROC curve shows how well the model distinguishes between classes.
% figure; 
% rocCurve(testTargets, testOutputs); 
% title('ROC Curve for Perceptron Diabetes Classifier');
