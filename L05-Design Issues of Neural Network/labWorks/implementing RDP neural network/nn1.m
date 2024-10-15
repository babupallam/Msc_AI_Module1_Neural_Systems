%% Non-Linearyly Seperable (NLS)
% Close all previous plots
close all;

% Define a problem with a set of nineteen 2-element input vectors P
% and the corresponding nineteen 1-element targets T.
P = [2 1 2 5 7 2 3 6 1 2 5 4 5 6 7 8 7 ;
     2 3 3 3 3 4 4 4 5 5 5 6 7 6 6 7 7];

T = [0 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0];

% Plot the initial data set (Non-Linear Separable (NLS) data)
figure(1);
plotpv(P, T);
%% Linearyly Seperable (LS)

% Now redefine P and T for a Linearly Separable (LS) subset of the NLS data
P = [2 1 2 5 7 2 3 6 1 2 5 4 5 6 7 8 7;
     2 3 3 3 3 4 4 4 5 5 5 6 7 6 6 7 7];

T = [0 0 0 1 1 0 0 1 0 0 1 1 1 1 1 1 1];

% Plot the updated LS data set
figure(2);
plotpv(P, T);

%%

% Calculate the minimum and maximum of each row (range for each input)
minMaxVal = minmax(P);

% Create a perceptron with one 2-element input and one output neuron
% Using default learning function (LEARNP)
net = newp(minMaxVal, 1);

% Simulate the network's output before training
simT_before = sim(net, P);

% Set the number of epochs for training to 100
net.trainParam.epochs = 100;

% Train the perceptron with the input data P and targets T
net = train(net, P, T);

% Simulate the network's output after training
simT_after = sim(net, P);

% Display the simulation results before and after training
disp('Output before training:');
disp(simT_before);

disp('Output after training:');
disp(simT_after);

% Plot the decision boundary of the trained perceptron
figure(2);
plotpc(net.iw{1, 1}, net.b{1});

%%
P = [P; simT_after ] ;
T = [ 0 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0 ] ;

%% 

% 3D classification

minMaxVal = minmax(P) ;
figure(3) ;
plotpv (P,T) ;

%%

% Create a Perceptron network
net = newp(minMaxVal, 1);

% Simulate the network's response to input data P
simT = sim(net, P);

% Set the number of training epochs
net.trainParam.epochs = 100;

% Train the network
net = train(net, P, T);

% Simulate the network's response to input data P after training
simT = sim(net, P);

%% 3D plot

figure(3)
plotpc(net.iw{1,1},net.b{1})

%%

% Create a Perceptron network
net = newp(minMaxVal, 1);

% Simulate the network's response to input data P
simT = sim(net, P);

% Set the number of training epochs
net.trainParam.epochs = 1000;

% Train the network
net = train(net, P, T);

% Simulate the network's response to input data P after training
simT = sim(net, P);

%% 3D plot

figure(3)
plotpc(net.iw{1,1},net.b{1})