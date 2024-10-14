%Creating RDP Neural Network

%close all previous plots
close all
%We define a problem, with a set of nineteen 
%2-element input vectors P and the corresponding nineteen 
%1-element targets T.
P = [2 1 2 5 7 2 3 6 1 2 5 4 6 5 6 7 8 7 8;
     2 3 3 3 4 4 4 5 5 5 6 6 7 6 6 7 6 7 7];
T = [0 0 0 1 0 0 1 0 0 1 1 1 1 0 0 0 0 0 0];
%Plot the data set
figure(1);
plotpv(P,T)

%Select an LS subset from within the NLS data set
P = [2 1 2 5 7 2 3 6 1 2 5 4 6 5 6 7 8 7 8;
     2 3 3 3 4 4 4 5 5 5 6 6 7 6 6 7 6 7 7];
T = [0 0 1 1 0 0 1 0 0 1 1 1 1 1 1 1 1 1 1];
%Plot the data set
figure(2);
plotpv(P,T)

%%

% Calculate the min and max of each column
minMaxVal = minmax(P);

% This code creates a perceptron layer with one 2-element
% input (ranges [0 1] and [-2 2]) and one neuron.
% (Supplying only two arguments to NEWP results in the default perceptron
% learning function LEARNP being used.)
net = newp(minMaxVal,1);

% Here we simulate the network's output, train for a
% maximum of 20 epochs, and then simulate it again.
simT = sim(net,P)
net.trainParam.epochs = 100;
net = train(net,P,T);
simT = sim(net,P)
figure(2);
plotpc(net.iw{1,1},net.b{1});


%%
% Append the new dimension to the input vector
P = [P; simT];
T = [0 0 0 1 1 0 0 1 0 0 1 1 1 1 1 1 0 0 0];

% Calculate the min and max values for the new 3D input data
minMaxVal = minmax(P);
% Plot the new data set in 3D space
figure(3);
plotpv(P, T);
title('3D Classification Problem');
xlabel('P(1)');
ylabel('P(2)');
zlabel('Simulated Output (3rd Dimension)');

% Create and train a new perceptron for the 3D problem
net = newp(minMaxVal, 1);
simT = sim(net, P)

net.trainParam.epochs = 10000;
net = train(net, P, T);
Y = sim(net, P);


figure(3)
plotpc(net.iw{1,1}, net.b{1});







