% Define the input matrix X and target vector T
% X represents two-dimensional input data for training, where each column corresponds to an input sample.
% Each row represents a feature (two features in this case).
X = [ -0.5 -0.5 +0.3 -0.1;  % First row represents the first input feature for each sample
      -0.5 +0.5 -0.5 +1.0]; % Second row represents the second input feature for each sample

% T represents the target outputs corresponding to the inputs in X.
% It is a binary classification problem where 1 represents one class and 0 represents another.
T = [1 1 0 0]; % Target output values for each input sample in X

% Plot the input data and target outputs in a 2D space.
% plotpv visualizes the points in the input space, showing their respective classes based on the target vector T.
plotpv(X, T);

%% Observation - 1 

% This section is another example of input data and targets, which has been commented out.
% If you uncomment it, you can observe how the perceptron behaves with a different dataset.

% X = [ -0.8 -0.5 -0.5 +0.3 -0.1;  % Modified input data for testing
%       -0.5 -0.5 +0.5 -0.5 +1.0]; % Modified second row of input features

% T = [0 1 1 0 0]; % Modified target output values

% plotpv(X, T); % Visualizes the new points in the input space with their respective classes

% Note: The above data points are not linearly separable, meaning a single straight line (decision boundary)
% cannot separate the classes perfectly. In such cases, the classical perceptron will fail to find a solution.

%%

% Create a perceptron neural network
net = perceptron; % The perceptron function in MATLAB initializes a single-layer, binary classification neural network, where:
%The network consists of one neuron (perceptron) in the output layer.
%The perceptron can learn a linear decision boundary to separate input data into two classes.

% Configure the perceptron network with the input data X and target T
% This step initializes the weights and biases based on the input data structure.
net = configure(net, X, T);

%%
% How configure() works?
%   X is a 2-by-4 matrix, meaning there are 2 input features and 4 training samples.
%   T is a 1-by-4 vector, representing 4 corresponding target values for each training sample.
%   The network's weights and biases are initialized randomly or set to default values.
%   The weights matrix (IW{1}) will have a size of 1-by-2, where:
%       1 represents the output neuron.
%       2 represents the number of input features.
%   The bias (b{1}) is a scalar value associated with the output neuron.

%   After executing net = configure(net, X, T);, the perceptron network net is now ready to be trained using the input data X and target values T.

%%

% Plot the input vectors and their corresponding target outputs again
plotpv(X, T); % Visualizes the input points on the graph

% Plot the decision boundary of the perceptron
% plotpc function is used to plot the classification line or hyperplane based on the perceptron's current weights and bias.
plotpc(net.IW{1}, net.b{1}); % IW{1} represents the weights and b{1} represents the bias of the perceptron's first layer

%%

% Create expanded versions of input and target sequences for adaptation
% con2seq converts the data into a sequential format required for training
% repmat replicates the data three times to create a larger dataset for training
XX = repmat(con2seq(X), 1, 3); % Create a sequence of the input data repeated three times
TT = repmat(con2seq(T), 1, 3); % Create a sequence of the target data repeated three times

% Here, 
%   X = [ -0.5 -0.5  0.3 -0.1;      -0.5  0.5 -0.5  1.0 ];
%   con2seq(X) = { [-0.5; -0.5], [-0.5; 0.5], [0.3; -0.5], [-0.1; 1.0] }
%   XX= repmat() = { [-0.5; -0.5], [-0.5; 0.5], [0.3; -0.5], [-0.1; 1.0], ...
%                [-0.5; -0.5], [-0.5; 0.5], [0.3; -0.5], [-0.1; 1.0], ...
%                [-0.5; -0.5], [-0.5; 0.5], [0.3; -0.5], [-0.1; 1.0] }
% Similarly,
%   TT = { [1], [1], [0], [0], ...
%          [1], [1], [0], [0], ...
%          [1], [1], [0], [0] }
%

% Adapt the perceptron using the replicated input and target data
% The adapt function allows the perceptron to update its weights and bias based on the training data
net = adapt(net, XX, TT); % Adapts the network with the new data to improve the classification

% Plot the updated decision boundary of the perceptron
plotpc(net.IW{1}, net.b{1}); 


%%

% Define a new input point to test the trained perceptron
x = [0.7; 1.2]; % New input vector to test the perceptron's prediction capability

% Calculate the output of the perceptron for the new input
y = net(x); % Pass the new input through the trained network to get the predicted output

% Plot the new input point in the input space
plotpv(x, y); % Visualizes the new input point along with its predicted output class

% Change the color of the plotted point to red for emphasis
point = findobj(gca, 'type', 'line'); % Find the most recently plotted object (the new input point)
point.Color = 'red'; % Set the color of the new input point to red to distinguish it from the training data

%%

% Hold the current plot to add more elements without erasing the existing plot
hold on;

% Re-plot the original input data and target outputs to provide context for the new point
plotpv(X, T); % Re-visualizes the original training data points
plotpc(net.IW{1}, net.b{1}); % Re-plots the decision boundary after the network's adaptation

% Release the hold on the current plot
hold off;
