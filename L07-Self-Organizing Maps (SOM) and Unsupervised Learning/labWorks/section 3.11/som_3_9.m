% Load sample data (customer purchasing behavior data)
data = randn(300, 4); % Randomly generated data with 300 samples and 4 features

% Display data information
disp('Sample customer purchasing behavior data generated:');
disp('First 5 rows of the data:');
disp(data(1:5, :)); % Display the first 5 rows to inspect the data

%%

% SOM training
net = selforgmap([10 10]); % Define a SOM with a 10x10 grid (100 neurons)
disp('SOM initialized with a 10x10 grid.');

net = train(net, data'); % Train the SOM using the transposed data
disp('SOM training completed.');

%%

% SOM Visualization: Plot neuron positions and data
figure;
plotsompos(net, data); % Visualize neuron positions with the data points
title('SOM Visualization: Customer Data Mining');
disp('SOM neuron positions visualized along with customer data.');

%%

% Visualizing the distribution of data for each feature
figure;
for i = 1:4
    subplot(2, 2, i); % Create a 2x2 grid of subplots
    histogram(data(:, i)); % Plot histogram of each feature
    xlabel(['Feature ', num2str(i)]);
    ylabel('Frequency');
    title(['Distribution of Feature ', num2str(i)]);
end
disp('Feature distribution histograms plotted.');

%%

% Visualizing clusters for selected feature pairs
figure;
gscatter(data(:,1), data(:,2), vec2ind(net(data')), 'rgb', 'osd'); % Color data based on SOM neuron assignment
xlabel('Feature 1');
ylabel('Feature 2');
title('SOM Clustering (Feature 1 vs Feature 2)');
disp('SOM clusters visualized for Feature 1 vs Feature 2.');

%%

