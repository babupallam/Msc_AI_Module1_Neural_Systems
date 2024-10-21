% Load the Iris dataset
% The 'meas' variable contains the features (sepal length, sepal width, petal length, petal width)
load fisheriris; 
data = meas; % Extract the data from the 'meas' variable

% Display a message indicating data loading success
disp('Fisher Iris dataset loaded successfully:');
disp('First five rows of data:');
disp(data(1:5, :)); % Display the first 5 rows for inspection

% Train SOM for clustering
dimension = [10 10]; % Set the dimensions for the SOM grid (10x10 neurons)
net = selforgmap(dimension); % Create a Self-Organizing Map (SOM) with specified grid dimensions

% Train the SOM using the input data
% Note: the data needs to be transposed (data') because the 'train' function expects observations as columns
net = train(net, data');

% Display the SOM architecture
view(net); % This will open a new window showing the SOM structure (grid of neurons)

%%
% Visualize the SOM clustering results
outputs = net(data'); % Obtain the SOM's response for each data point (which neuron it was mapped to)

% Display the clustering outputs
disp('SOM output (neuron assignment for each data point):');
disp(outputs);
%%
% Visualize the positions of the neurons in relation to the input data
figure;
plotsompos(net, data); % Visualize SOM neuron positions and how they map to data points
title('SOM Clustering Visualization');

%%
% Additional Visualization: Color the SOM neurons according to the class labels
% The Fisher Iris dataset has 3 known classes, stored in the 'species' variable
% Let's visualize which neurons are associated with each class
species = grp2idx(species); % Convert species names into numeric labels (1, 2, 3)

% Map the class labels to the corresponding SOM neuron for each data point
% We will use gscatter to create a scatter plot with colors representing the classes
figure;
gscatter(data(:,1), data(:,2), species); % Plot Sepal Length vs Sepal Width colored by class
title('Original Data (Sepal Length vs Sepal Width) Colored by Class');

%%
% Visualize SOM classification for comparison
% We'll use the neuron assignments from SOM (outputs) to color the data points
% and compare with the actual class labels
neuronIndex = vec2ind(outputs); % Convert neuron outputs to indices (i.e., the neuron each data point was mapped to)

figure;
gscatter(data(:,1), data(:,2), neuronIndex); % Plot Sepal Length vs Sepal Width colored by neuron
title('SOM Neuron Mapping of Data (Sepal Length vs Sepal Width)');
