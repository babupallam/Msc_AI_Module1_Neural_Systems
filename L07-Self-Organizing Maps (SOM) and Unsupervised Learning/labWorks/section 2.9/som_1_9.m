% Load the Iris dataset
load fisheriris; 
data = meas; % Extract the features (sepal length, sepal width, petal length, petal width)

%%
% Train SOM
net = selforgmap([10 10]); % Create a 10x10 SOM grid
net = train(net, data'); % Train the SOM on the transposed data (observations as columns)

%%

% Plot U-Matrix
figure;
plotsom(net.layers{1}.positions); % Visualize the neurons' grid positions
title('SOM U-Matrix Visualization');


%%