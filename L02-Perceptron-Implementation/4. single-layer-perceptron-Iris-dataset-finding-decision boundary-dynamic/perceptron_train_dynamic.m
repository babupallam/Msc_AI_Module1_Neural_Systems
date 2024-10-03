% File: perceptron_train_dynamic.m
% Purpose: Train a single-layer perceptron and visualize the decision boundary dynamically

function weights = perceptron_train_dynamic(X, y, learning_rate, epochs)
    % X: Input features matrix (each row is a sample, each column a feature)
    % y: Labels (1 or -1 for each sample)
    % learning_rate: The step size for weight updates
    % epochs: Number of times to iterate over the entire dataset
    % weights: The learned weights after training

    [n_samples, n_features] = size(X);
    
    % Initialize weights (including bias as the last weight)
    weights = zeros(n_features + 1, 1);
    
    % Add bias term to the input matrix (last column all ones)
    X = [X, ones(n_samples, 1)];

    % Set up the figure for dynamic visualization
    figure;
    hold on;
    
    % Plot initial data points (Class 1: Setosa, Class 2: Versicolor)
    h1 = scatter(X(y == 1, 1), X(y == 1, 2), 'bo', 'filled'); % Class 1 (Setosa)
    h2 = scatter(X(y == -1, 1), X(y == -1, 2), 'ro', 'filled'); % Class 2 (Versicolor)

    % Axis limits for consistent visualization
    x_vals = min(X(:, 1)) : 0.01 : max(X(:, 1));
    ylim([min(X(:, 2)) max(X(:, 2))]);

    % Initialize plot for the decision boundary (empty line that will be updated)
    h_boundary = plot(x_vals, zeros(size(x_vals)), 'k-', 'LineWidth', 2);

    % Training loop with dynamic boundary updates
    for epoch = 1:epochs
        for i = 1:n_samples
            % Predict output using current weights
            prediction = sign(X(i, :) * weights);
            
            % Update weights if the prediction is incorrect
            if prediction ~= y(i)
                % Update rule: weights = weights + learning_rate * error * input
                weights = weights + learning_rate * y(i) * X(i, :)';

                 % Print the updated weights after each update
                fprintf('Epoch %d, Sample %d: Updated Weights = [%.4f, %.4f, %.4f]\n', ...
                epoch, i, weights(1), weights(2), weights(3));
          
            end
        end
        
        % Update decision boundary after each epoch
        y_vals = -(weights(1) * x_vals + weights(3)) / weights(2);
        set(h_boundary, 'YData', y_vals); % Update boundary line

        % Update the plot title to show the current epoch
        title(sprintf('Perceptron Decision Boundary - Epoch %d', epoch));
        xlabel('Feature 1');
        ylabel('Feature 2');
        legend([h1, h2, h_boundary], 'Setosa', 'Versicolor', 'Decision Boundary');

        % Pause to visualize the update
        pause(0.5);
    end
    
    hold off;
end
