% We define a problem, with a set of fourteen 2-element input vectors P
% and the corresponding fourteen 1-element targets T.
P = [2 1 2 5 7 2 3 6 1 2 5 4 6 5; 2 3 3 3 3 4 4 4 5 5 5 6 6 7];
T = [0 0 0 1 1 0 0 1 0 0 1 1 1 1];

% Plot the data set
figure(1);
plotpv(P, T);

hold on; % add the hold on the plot

%%

% Create the Perceptron Neural Network
% Calculate the min and max of each column
minMaxVal = minmax(P);

% This code creates a perceptron layer with one 2-element input
% and one neuron. Supplying only two arguments to NEWP results in the 
% default perceptron learning function LEARNP being used.
net = newp(minMaxVal, 1);

% Configure the training parameters
net.trainParam.epochs = 20; % Maximum epochs for training
net.trainParam.goal = 0; % Performance goal
net.trainParam.showWindow = true; % Disable the training GUI to avoid errors

% Train the network and capture the training record
[net, tr] = train(net, P, T);
disp(['Training stopped due to: ', tr.stop]);


% Check if the performance goal was met
if tr.best_perf <= net.trainParam.goal
    disp('TRAINC, Performance goal met');
elseif tr.num_epochs >= net.trainParam.epochs
    disp('TRAINC, Maximum epoch reached');
else
    disp('TRAINC, Training stopped due to other reasons');
end



% Simulate the network's output before and after training
simT_before = sim(net, P); % Before training
disp('Output before training:');
disp(simT_before);

% Train the network with the data (retraining for additional epochs)
net.trainParam.epochs = 20; % Set maximum number of training epochs to 20
net = train(net, P, T);

% Simulate the network again after training
simT_after = sim(net, P); % After training
disp('Output after training:');
disp(simT_after);

%%

% Plot the decision boundary to visualize the trained network
figure(1); % Ensure figure 1 is active
plotpc(net.IW{1}, net.b{1}); % Plot the perceptron's decision boundary


%%

% Now extend the training to 200 epochs
net.trainParam.epochs = 200; % Set maximum number of training epochs to 200
[net, tr] = train(net, P, T);

% Plot the updated decision boundary after additional training
figure(2); % Create a new figure for the updated decision boundary
plotpv(P, T);
hold on; % Keep the plot open for the updated decision boundary
plotpc(net.IW{1}, net.b{1}); % Plot the updated decision boundary
title('Updated Decision Boundary after Training for 200 Epochs');
hold off; % Release the hold on the plot

%%