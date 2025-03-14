function [w_estimate,J_history] = Iterative_Scaling1(X,y,max_iters,d)
    % Logistic regression using iterative scaling version 1

    % Initialization
    w_estimate = zeros(d,1);
    J_history = zeros(max_iters,1);
    
    % Iterative Scaling Algorithm
    for iter = 1:max_iters
        % Calculate s
        s = max(sum(abs(X), 2));
        
        % Update each parameter w_k
        for k = 1:d
            % Calculate the numerator and denominator for the update
            numerator = sum((1 - sigmoid(y.*(X*w_estimate))).*abs(X(:,k)).*(y.*X(:,k)>0));
            denominator = sum((1 - sigmoid(y.*(X*w_estimate))).*abs(X(:,k)).*(y.*X(:,k)<0));
            
            % Update the parameter w_k
            w_estimate(k) = w_estimate(k) + (1/(2*s))*log(numerator/denominator);
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