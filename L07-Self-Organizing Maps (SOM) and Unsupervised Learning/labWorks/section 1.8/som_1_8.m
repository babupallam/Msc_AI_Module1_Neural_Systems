% Load the Iris dataset
% The 'meas' variable contains the feature data: sepal length, sepal width, petal length, and petal width
load fisheriris; 
data = meas; % Extract the feature data for training the SOM

% Display a message indicating that the data has been loaded successfully
disp('Fisher Iris dataset loaded successfully:');
disp('First five rows of data:');
disp(data(1:5, :)); % Display the first 5 rows of the dataset

%%

% Initialize a Self-Organizing Map (SOM)
% Define the SOM grid size as 10x10 neurons, which will cluster the data
som_dimension = [10 10]; 
net = selforgmap(som_dimension); % Create a SOM with the specified grid dimensions

% Train the SOM on the Iris data
% The data is transposed because the SOM expects input vectors as columns
net = train(net, data');

% Display the SOM architecture in a separate window
view(net);

%%
% Visualize the SOM Neuron Positions
% This plot shows the positions of the neurons in relation to the input data
figure;
plotsompos(net, data); 
title('SOM Neuron Position');

%%
% Display the output SOM response (neuron assignment for each data point)
outputs = net(data'); % Get the SOM's response for each data point (which neuron it's assigned to)

% Display the neuron assignments for the first 5 data points
disp('SOM Neuron assignments for the first five data points:');
disp(outputs(:, 1:5)); % Display assignments for the first 5 data points

%%
% Additional Visualization: Compare SOM clustering with true Iris species labels

% Convert the species names (setosa, versicolor, virginica) to numeric labels (1, 2, 3)
species = grp2idx(species); 

% Scatter plot to visualize the data using the first two features (Sepal Length and Sepal Width)
figure;
gscatter(data(:,1), data(:,2), species, 'rgb', 'osd'); % Color the points based on species
title('Original Data (Sepal Length vs Sepal Width) Colored by Species');

% SOM-based clustering visualization using neuron assignments
% We'll convert the SOM output (neuron responses) into neuron indices
neuronIndex = vec2ind(outputs); % Convert the neuron responses into integer neuron indices

figure;
gscatter(data(:,1), data(:,2), neuronIndex, 'rgb', 'osd'); % Color the points based on SOM neuron assignment
title('SOM Neuron Mapping of Data (Sepal Length vs Sepal Width)');

% Side-by-side comparison of true species labels and SOM neuron assignment
figure;
subplot(1, 2, 1); % Plot 1: True species labels
gscatter(data(:,1), data(:,2), species, 'rgb', 'osd'); % Visualize Sepal Length vs Sepal Width by species
title('True Species Labels');

subplot(1, 2, 2); % Plot 2: SOM neuron assignment
gscatter(data(:,1), data(:,2), neuronIndex, 'rgb', 'osd'); % Visualize Sepal Length vs Sepal Width by neuron index
title('SOM Neuron Assignment');
