% Perceptron Learning Algorithm for AND operation with explicit bias handling

% Define the input matrix X and target vector T for the AND operation
X = [0 0; 0 1; 1 0; 1 1]; % Input data (each row is an input sample)
T = [0 0 0 1];            % Target outputs corresponding to AND operation

% Initialize weights and bias
weights = [0 0];  % Initial weights for x1 and x2
bias = -1;        % Initial bias
learning_rate = 1; % Learning rate
max_epochs = 100; % Maximum number of epochs to prevent infinite loops

% Initialize variables for tracking updates
num_updates = 0;
converged = false;

% Perceptron learning algorithm
for epoch = 1:max_epochs
    errors = 0; % Counter for misclassifications in the current epoch
    
    % Iterate through all samples in the dataset
    for i = 1:size(X, 1)
        % Extract the current input and target output
        x = X(i, :);
        target = T(i);
        
        % Calculate the weighted sum (z) including the bias
        z = dot(weights, x) + bias;
        
        % Apply the activation function (step function)
        if z >= 0
            output = 1;
        else
            output = 0;
        end
        
        % Update weights and bias if there is a misclassification
        if output ~= target
            % Calculate the update value
            update = learning_rate * (target - output);
            
            % Update weights and bias
            weights = weights + update * x;
            bias = bias + update;
            
            % Increment the error counter and number of updates
            errors = errors + 1;
            num_updates = num_updates + 1;
            
            % Display the update information
            fprintf('Update %d: Weights = [%.2f, %.2f], Bias = %.2f\n', ...
                num_updates, weights(1), weights(2), bias);
        end
    end
    
    % If no errors occurred in this epoch, the algorithm has converged
    if errors == 0
        fprintf('Converged after %d epochs with %d total updates.\n', epoch, num_updates);
        converged = true;
        break;
    end
end

% If the algorithm did not converge within the maximum number of epochs
if ~converged
    fprintf('Did not converge after %d epochs. Final Weights = [%.2f, %.2f], Bias = %.2f\n', ...
        max_epochs, weights(1), weights(2), bias);
end

% Test the trained perceptron on the AND dataset
fprintf('\nTesting the trained perceptron:\n');
for i = 1:size(X, 1)
    x = X(i, :);
    target = T(i);
    
    % Calculate the weighted sum (z) including the bias
    z = dot(weights, x) + bias;
    
    % Apply the activation function (step function)
    if z >= 0
        output = 1;
    else
        output = 0;
    end
    
    % Display the result
    fprintf('Input = [%d, %d], Target = %d, Predicted = %d\n', ...
        x(1), x(2), target, output);
end
