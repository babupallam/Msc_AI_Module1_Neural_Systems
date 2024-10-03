% TEST_MSE_LOSS: Script to test the MSE loss function.
% This script provides example true and predicted values, and calculates 
% the Mean Squared Error between them.

% Define the true values (targets) for the test
y_true = [0.5, 0.1];  % Example true output values

% Define the predicted values from a neural network
y_pred = [0.4, 0.2];  % Example predicted values from the model

% Call the mse_loss function to compute the loss
loss = mse_loss(y_true, y_pred);

% Display the result to the user
disp(['MSE Loss: ', num2str(loss)]);  % Print the MSE value
