%% BENCHMARK A
% Close all previous plots that might be open
close all

% Define a problem with a set of 19 two-element input vectors P
% and the corresponding 19 target values T.

% 'P' is a 2x19 matrix where each column represents a 2-element input vector.
% The first row holds the x-coordinates, and the second row holds the y-coordinates.
P = [ 2 1 2 5 7 2 3 6 1 2 5 4 5 6 7 8 7;
      2 3 3 3 3 4 4 4 5 5 5 6 7 6 6 7 7 ];

% 'T' is a 1x19 vector of target values corresponding to each input vector in P.
% It contains binary values (0 or 1) representing class labels for each point.
T = [ 0 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0 ];

figure(1);   % Open a new figure window for the plot
plotpv(P,T); % Plot the input vectors P and their corresponding targets T

% The same input vectors P are redefined (which in this case is identical 
% to the previous P), but the target values T are modified for a second scenario.

P = [ 2 1 2 5 7 2 3 6 1 2 5 4 5 6 7 8 7 ;
      2 3 3 3 3 4 4 4 5 5 5 6 7 6 6 7 7 ];

% Here, 'T' is updated, and the last few elements are changed to 1.
T = [ 0 0 0 1 1 0 0 1 0 0 1 1 1 1 1 1 1 ];

% Plot the updated dataset with the new targets in a second figure window.
figure(2);   % Open a new figure window
plotpv(P,T); % Plot the new dataset with updated targets



%% BENCHMARK B

% Calculate the minimum and maximum of each row (each feature) in the matrix P.
% The function 'minmax' returns a 2x2 matrix where each column contains the 
% minimum and maximum values for each row of P. 
% The result is used to define the range of inputs to the neural network.
minMaxVal = minmax(P);
disp(minMaxVal) % from the above [1 8; 2 7]


% This code creates a perceptron layer with one 2-element input vector and one neuron.
% The input vectors are limited to the ranges defined in 'minMaxVal'.
% - 'newp(minMaxVal, 1)' initializes a perceptron network.
% - The first argument, 'minMaxVal', defines the range of inputs.
% - The second argument, '1', specifies that the network will have one neuron in its output layer.
% - Since no learning function is specified, it defaults to 'learnp', a standard perceptron learning function.
net = newp(minMaxVal, 1); % 2 input neuron and 1 input output

% Simulate the initial output of the network before training.
% Simulate the perceptron's output using the current weights and input 'P'
simT = sim(net, P);

% Display the message followed by the simulated output
disp("Before Training:");
disp(simT);

% Set the maximum number of epochs (training iterations) to 100.
% This specifies that the network will train for a maximum of 100 passes over the data.
net.trainParam.epochs = 100;

% Train the perceptron network with the input vectors P and target values T.
% 'train(net, P, T)' adjusts the network weights based on the training data.
net = train(net, P, T);

% Simulate the network's output after training.
% After the network has been trained, 'sim' computes the output based on the learned weights.
simT = sim(net, P);
disp("After Training: ")
disp(simT)


% Plot the classification line created by the perceptron.
% 'plotpc' visualizes the decision boundary learned by the perceptron.
% - 'net.iw{1,1}' contains the weights of the input-to-output connections.
% - 'net.b{1}' contains the bias term of the neuron.
figure(2); % Open a new figure window (this will overwrite the previous figure(2)).
plotpc(net.iw{1,1}, net.b{1});

%% BENCHMARK C

% This is where RDP implements.. make 2D to 3D by elivating the non linear
% subset 
% Append the simulated output 'simT' to the matrix 'P'.
% 'P' originally contains 2-element input vectors (2x18 matrix).
% After appending 'simT' (1x18 row vector) to 'P', the new 'P' becomes a 3x18 matrix.
% This means that 'P' now has three rows, where:
% - Rows 1 and 2 represent the original input features (x and y coordinates).
% - Row 3 represents the simulated output (predicted classification values by the perceptron) for each input.
P = [P; simT];

% Reassign the target vector 'T' to its original form.
% This is resetting 'T' to its previous state, ensuring that it corresponds to the original target values.
T = [0 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0];


% Calculate the minimum and maximum values for each row (dimension) in the updated matrix 'P'.
% 'P' now has 3 rows: two for the input features (x, y) and one for the simulated output (predicted classes).
% 'minmax' will calculate the min and max for each row and return a 2x3 matrix, 
% where each column contains the min and max of each row in 'P'.
minMaxVal = minmax(P);

% Open a new figure window (Figure 3) for plotting the updated data.
figure(3);

% Plot the input data points with their targets using 'plotpv'.
plotpv(P, T);

%% BENCHMARK D
% Create a new perceptron network with one neuron.
% 'minMaxVal' is a 2x3 matrix that defines the input ranges for the perceptron.
% Since 'P' now has 3 rows, each column of 'minMaxVal' represents the range of 
% values for each feature (the first two input features and the third row which contains the previous simulated outputs).
net = newp(minMaxVal, 1);

% Simulate the output of the untrained network using the updated input data 'P'.
% 'sim' runs the perceptron with its initial random weights, producing output based on the input 'P'.
simT = sim(net, P);

% Set the maximum number of training epochs (iterations) to 100.
% 'trainParam.epochs' is a property of the network that determines how many times
% the network will iterate over the training data during training.
net.trainParam.epochs = 100;

% Train the network using the input data 'P' and the target values 'T'.
% The network will adjust its weights and bias to reduce classification errors.
net = train(net, P, T);

% Simulate the output of the trained network with the input 'P'.
% After training, the network is expected to produce more accurate predictions based on the learned weights.
simT = sim(net, P);


% Update the figure (3) with the plane
% Open a new figure window (Figure 3) for plotting the perceptron's decision boundary.
figure(3);

% Plot the decision boundary of the trained perceptron using 'plotpc'.
% 'net.iw{1,1}' contains the input weights of the network after training.
% 'net.b{1}' contains the bias term of the neuron after training.
% 'plotpc' plots the linear decision boundary that the perceptron has learned.
plotpc(net.iw{1,1}, net.b{1});


%% BENCHMARK E

% Create a new perceptron network with one neuron.
% 'minMaxVal' is a 2x3 matrix that defines the input ranges for the perceptron.
% Since 'P' now has 3 rows, each column of 'minMaxVal' represents the range of 
% values for each feature (the first two input features and the third row which contains the previous simulated outputs).
net = newp(minMaxVal, 1);

% Simulate the output of the untrained network using the updated input data 'P'.
% 'sim' runs the perceptron with its initial random weights, producing output based on the input 'P'.
simT = sim(net, P);

% Set the maximum number of training epochs (iterations) to 1000.
% 'trainParam.epochs' is a property of the network that determines how many times
% the network will iterate over the training data during training.
net.trainParam.epochs = 1000;

% Train the network using the input data 'P' and the target values 'T'.
% The network will adjust its weights and bias to reduce classification errors.
net = train(net, P, T);

% Simulate the output of the trained network with the input 'P'.
% After training, the network is expected to produce more accurate predictions based on the learned weights.
simT = sim(net, P);


% Update the plane again with new plane

% Open a new figure window (Figure 3) for plotting the perceptron's decision boundary.
figure(3);

% Plot the decision boundary of the trained perceptron using 'plotpc'.
% 'net.iw{1,1}' contains the input weights of the network after training.
% 'net.b{1}' contains the bias term of the neuron after training.
% 'plotpc' plots the linear decision boundary that the perceptron has learned.
plotpc(net.iw{1,1}, net.b{1});

%%
