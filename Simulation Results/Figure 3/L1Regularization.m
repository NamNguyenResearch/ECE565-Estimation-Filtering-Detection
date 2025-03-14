function [w_estimate, J_history] = L1Regularization(X,y,alpha,lambda,max_iters)
    % Logistic regression using gradient descent with L1-regularization
    
    w_estimate = zeros(size(X,2),1);
    J_history = zeros(max_iters,1);

    for iter = 1:max_iters
        % Compute the hypothesis
        h = 1./(1+exp(-X*w_estimate));

        % Compute the cost function with L1-regularization
        J_history(iter) = (-1/length(y))*sum(y.*log(h)+(1-y).*log(1-h))+ ...
                          (lambda/length(y))*sum(abs(w_estimate));

        % Compute the gradient with L1-regularization
        gradient(1) = (1/length(y))*X(:,1)'*(h-y);
        gradient(2:end) = (1/length(y))*(X(:,2:end)'*(h-y)+lambda*sign(w_estimate(2:end)));

        % Update parameters using gradient descent
        w_estimate = w_estimate - alpha*gradient;
    end
end