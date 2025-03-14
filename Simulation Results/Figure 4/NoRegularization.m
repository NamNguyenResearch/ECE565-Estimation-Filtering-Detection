function [w_estimate,J_history] = NoRegularization(X,y,alpha,max_iters)
    % Logistic regression using gradient descent
    
    w_estimate = zeros(size(X,2),1);
    J_history = zeros(max_iters,1);

    for iter = 1:max_iters
        % Compute the hypothesis
        h = 1./(1+exp(-X*w_estimate));
        
        % Compute the cost function
        J_history(iter) = (-1/length(y))*sum(y.*log(h)+(1-y).*log(1-h));
        
        % Compute the gradient
        gradient = (1/length(y))*X'*(h-y);
        
        % Update parameters using gradient descent
        w_estimate = w_estimate - alpha*gradient;
    end
end

