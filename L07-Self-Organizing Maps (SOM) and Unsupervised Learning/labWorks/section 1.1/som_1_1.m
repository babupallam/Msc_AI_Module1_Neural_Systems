%% Supervised Learning Example using Support Vector Machine (SVM)

% Generate random data for two classes:
% Class 1 centered around (1,1), and Class 2 centered around (-1,-1)
X = [randn(100,2)+1; randn(100,2)-1];

% Define the labels: 
% Class 1: label 1 for the first 100 points
% Class 2: label -1 for the second 100 points
Y = [ones(100,1); -ones(100,1)]; 

% Train a Support Vector Machine (SVM) classifier
SVMModel = fitcsvm(X, Y);

% Display SVM model details
disp('SVM Model Trained:');
disp(SVMModel);

% Visualize the data points
figure;
gscatter(X(:,1), X(:,2), Y, 'rb', 'xo'); % Scatter plot for the two classes
hold on;

% Plot the decision boundary
% Create a grid of points to evaluate the SVM decision function
[x1Grid, x2Grid] = meshgrid(min(X(:,1)):0.1:max(X(:,1)), min(X(:,2)):0.1:max(X(:,2)));
XGrid = [x1Grid(:), x2Grid(:)];

% Predict the classes for each point in the grid
[~, scores] = predict(SVMModel, XGrid);

% Plot the decision boundary and margins
contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [0 0], 'k'); % Decision boundary
contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [-1 1], '--k'); % Margins
title('Supervised Learning: SVM Decision Boundary');
hold off;

%% Unsupervised Learning Example using k-means Clustering

% Generate random data for clustering
data = [randn(100,2)+1; randn(100,2)-1];

% Apply k-means clustering to group the data into 2 clusters
idx = kmeans(data, 2);

% Display clustering results
disp('k-means clustering results:');
disp('Cluster assignment for each data point:');
disp(idx);

% Visualize the clusters
figure;
gscatter(data(:,1), data(:,2), idx, 'rb', 'xo'); % Color points based on cluster assignment
title('Unsupervised Learning: Clustering with k-means');

% Additional Visualization: Compare true labels with k-means clustering
% Generate true labels for comparison (same structure as in SVM example)
trueLabels = [ones(100,1); -ones(100,1)];

% Visualize the true labels as a scatter plot
figure;
gscatter(data(:,1), data(:,2), trueLabels, 'rb', 'xo'); % True classes plot
title('Original Data with True Labels');

% Visualize the k-means clusters alongside the true labels for comparison
figure;
subplot(1, 2, 1); % Plot 1: Clustering results
gscatter(data(:,1), data(:,2), idx, 'rb', 'xo');
title('k-means Clustering Results');

subplot(1, 2, 2); % Plot 2: True labels
gscatter(data(:,1), data(:,2), trueLabels, 'rb', 'xo');
title('True Labels vs. k-means Clustering');
