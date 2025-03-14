function [w_estimate,J_history] = Iterative_Scaling2(X,y,max_iters,d)
    % Logistic regression using iterative scaling version 2

    % Initialization
    w_estimate = zeros(d,1);
    J_history = zeros(max_iters,1);
    
    % Iterative Scaling Algorithm
    for iter = 1:max_iters
        % Calculate S and initialize variables
        S = max(sum(abs(X), 2));
        A1 = zeros(size(X, 2), 1);
        A2 = zeros(size(X, 2), 1);
        B = X' * y;
        
        % Calculate A1 and A2
        for k = 1:size(X, 2)
            A1(k) = 0.5 * sum((abs(X(:, k)) + X(:, k)) .* sigmoid(X * w_estimate) .* y);
            A2(k) = 0.5 * sum((abs(X(:, k)) - X(:, k)) .* sigmoid(X * w_estimate) .* y);
        end
        
        % Update each parameter w_k
        for k = 1:size(X, 2)
            % Calculate the argument of the logarithm
            argument_log = B(k) + sqrt(B(k)^2 + 4 * A1(k) * A2(k));
            
            % Update the parameter w_k
            w_estimate(k) = w_estimate(k) + (1 / S) * log(argument_log / (2 * A1(k)));
        end

        % Compute the hypothesis
        h = 1./(1+exp(-X*w_estimate));
        
        % Compute the cost function
        J_history(iter) = (-1/length(y))*sum(y.*log(h)+(1-y).*log(1-h));
    end
end

% The sigmoid function
function output = sigmoid(x)
    output = 1./(1 + exp(-x));
end