% Define a problem with a set of fourteen 2-element input vectors P
% and the corresponding fourteen 1-element targets T.
%P = [2 1 2 5 7 2 3 6 1 2 5 4 6 5 6 7 8 7;
%     2 3 3 3 3 4 4 4 5 5 5 6 6 7 6 6 7 7];
%T = [0 0 0 1 1 0 0 1 0 0 1 1 1 1 0 0 0 0];

% removed one pair from the above
P = [2 1 2 5 7 2 3 6 1 2 5 4 5 6 7 8 7;
     2 3 3 3 3 4 4 4 5 5 5 6 7 6 6 7 7];
T = [0 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0];

% Plot the data set
figure(1);
plotpv(P, T);


%%
% Define the input vectors P and target values T

P = [2 1 2 5 7 2 3 6 1 2 5 4 5 6 7 8 7;
     2 3 3 3 3 4 4 4 5 5 5 6 7 6 6 7 7];
T = [0 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0];

% Calculate the minimum and maximum of each row (range for each input)
minMaxVal = minmax(P);

% Create a perceptron with one 2-element input and one output neuron
% Using default learning function (LEARNP)
net = newp(minMaxVal, 1);

% Simulate the network's output before training
simT_before = sim(net, P);

% Train the network for a maximum of 20 epochs
net.trainParam.epochs = 20;
net = train(net, P, T);

% Simulate the network's output after training
simT_after = sim(net, P);

% Display the results
disp('Output before training:');
disp(simT_before);

disp('Output after training:');
disp(simT_after);

%%
figure(1) ;
plotpc(net.iw{1,1},net.b{1})

%% train the above code for more nummber of epoch
disp("Performing for more number of epoches: ")
% Train the network for a maximum of 1000 epochs
net.trainParam.epochs = 1000;
net = train(net, P, T);

% Simulate the network's output after training
simT_after = sim(net, P);

% Display the results
disp('Output before training:');
disp(simT_before);

disp('Output after training:');
disp(simT_after);


% Plot the data set
figure(2);
plotpv(P, T);
figure(2) ;
plotpc(net.iw{1,1},net.b{1})
%%


% Observations:
%  - here, all the solutions in class 0 has been classified currectly (see the
%  output diagrams)