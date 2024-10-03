function loss = mse_loss(y_true, y_pred)
    % MSE_LOSS: This function calculates the Mean Squared Error (MSE) loss.
    % MSE is a common loss function used in regression problems.
    %
    % Inputs:
    % y_true - A vector of true (target) values
    % y_pred - A vector of predicted values (from the network)
    %
    % Output:
    % loss - The calculated MSE value
    
    % Step 1: Compute the squared differences between the true and predicted values
    squared_errors = (y_true - y_pred).^2;
    
    % Step 2: Compute the mean of these squared differences
    loss = mean(squared_errors);  % Mean Squared Error formula
end
