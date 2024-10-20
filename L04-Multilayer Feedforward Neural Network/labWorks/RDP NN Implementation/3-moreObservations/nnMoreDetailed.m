% Close all previous plots that might be open
close all

% Define a problem with a set of 19 two-element input vectors P
% and the corresponding 19 target values T.
% 'P' is a 2x19 matrix where each column represents a 2-element input vector.
P = [ 2 1 2 5 7 2 3 6 1 2 5 4 5 6 7 8 7;
      2 3 3 3 3 4 4 4 5 5 5 6 7 6 6 7 7 ];

% 'T' is a 1x19 vector of target values corresponding to each input vector in P.
T = [ 0 0 0 1 1 0 0 1 0 0 1 1 1 1 1 1 1 ];tes


% Plot the dataset using 'plotpv' to visualize initial data points
figure(1);
plotpv(P, T);

%%
% Calculate the minimum and maximum of each row (each feature) in the matrix P.
minMaxVal = minmax(P);
disp('Input range for original input features:');
disp(minMaxVal);


% Create and initialize a perceptron network
net = newp(minMaxVal, 1);

% Simulate the initial output of the perceptron before training
simT = sim(net, P);
disp('Before Training:');
disp(simT);

% Set the maximum number of epochs for training
net.trainParam.epochs = 100;

% Train the perceptron network with the original input vectors P and target values T
net = train(net, P, T);

% Simulate the network's output after training
simT = sim(net, P);
disp('After Training:');
disp(simT);

% Plot the classification line created by the perceptron
figure(1);
plotpc(net.iw{1,1}, net.b{1});

%%
% RDP Implementation: Expanding input space to make NLS problem LS
% Append the simulated output 'simT' to the matrix 'P', creating a 3D input space
P = [P; simT];

% Reassign the target vector 'T' to its original form
T = [ 0 0 0 1 1 0 0 1 0 0 1 1 1 1 1 1 1 ];

% Calculate the minimum and maximum values for each row (dimension) in the updated matrix 'P'
minMaxVal = minmax(P);
disp('Input range for expanded input features:');
disp(minMaxVal);

% Create a new perceptron network with the expanded input space
net = newp(minMaxVal, 1);

% Simulate the output of the untrained network using the updated input data 'P'
simT = sim(net, P);
disp('Before Training with Expanded Input Space:');
disp(simT);

% Set the maximum number of training epochs for training with the expanded input space
net.trainParam.epochs = 100;

% Train the network using the updated input data 'P' and the target values 'T'
net = train(net, P, T);

% Simulate the output of the trained network with the updated input 'P'
simT = sim(net, P);
disp('After Training with Expanded Input Space:');
disp(simT);

% Plot the updated data points and decision boundary in a new figure
figure(3);
plotpv(P, T);
plotpc(net.iw{1,1}, net.b{1});

%%
% Train the above model with 1000 epoches

% Create another perceptron network for the further expanded input space
net = newp(minMaxVal, 1);

% Set the maximum number of training epochs to 1000 for a more comprehensive learning process
net.trainParam.epochs = 1000;

% Train the perceptron network with the further expanded input data
net = train(net, P, T);

% Simulate the network's output after training with the further expanded input space
simT = sim(net, P);
disp('After Training with Further Expanded Input Space:');
disp(simT);

% Update figure (3) with the final decision boundary
figure(3);
plotpc(net.iw{1,1}, net.b{1});