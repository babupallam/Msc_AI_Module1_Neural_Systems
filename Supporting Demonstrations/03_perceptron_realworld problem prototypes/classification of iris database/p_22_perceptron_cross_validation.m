% p_22_perceptron_cross_validation.m
% Implementation of perceptron with 5-fold cross-validation
% Task: Classify iris flowers into different species using a perceptron and evaluate using cross-validation

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

% Normalize the input features
inputs = normalize(inputs); % Normalize the features to improve training performance

%% Section 2: Apply 5-fold cross-validation
% Split the data into 5 folds and train/test in each fold
k = 5;
cv = cvpartition(binaryTargets, 'KFold', k);
accuracies = zeros(1, k); % To store accuracy for each fold

for i = 1:k
    % Training and testing indices for this fold
    trainInd = training(cv, i);
    testInd = test(cv, i);
    
    % Prepare training and testing data
    trainInputs = inputs(:, trainInd);   % Training input features
    trainTargets = binaryTargets(trainInd);  % Training labels
    testInputs = inputs(:, testInd);     % Test input features
    testTargets = binaryTargets(testInd);    % Test labels
    
    % Ensure targets are row vectors
    trainTargets = trainTargets(:)';     
    testTargets = testTargets(:)';
    
    % Train the perceptron
    net = perceptron;
    net = train(net, trainInputs, trainTargets); % Train the perceptron
    
    % Test the perceptron on test data
    testOutputs = net(testInputs); % Get predictions
    testOutputs = round(testOutputs); % Round to binary (0 or 1)
    
    % Calculate accuracy for this fold
    accuracies(i) = sum(testOutputs == testTargets) / length(testTargets) * 100; % Calculate accuracy
end

%% Section 3: Display results
% Display the accuracy of each fold
for i = 1:k
    disp(['Fold ', num2str(i), ' Accuracy: ', num2str(accuracies(i)), '%']);
end

% Calculate and display the mean accuracy across all folds
meanAccuracy = mean(accuracies);
disp(['Mean Accuracy across ', num2str(k), '-fold cross-validation: ', num2str(meanAccuracy), '%']);
