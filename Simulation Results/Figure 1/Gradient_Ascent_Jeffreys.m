function [w_estimate, J_history] = Gradient_Ascent_Jeffreys(X, y, max_iters, alpha_Jeffreys, sigma_2)
    % Initialization
    w_estimate = zeros(size(X, 2), 1);
    J_history = zeros(max_iters, 1);

    % Repeat until convergence
    for iter = 1:max_iters
        % Compute the gradient
        gradient = compute_gradient(X, y, w_estimate, sigma_2);

        % Update weights
        w_estimate = w_estimate + alpha_Jeffreys * gradient;

        % Compute the hypothesis
        h = 1./(1 + exp(-X * w_estimate));

        % Compute the cost function
        J_history(iter) = compute_cost(y, h);
    end
end

function gradient = compute_gradient(X, y, w_estimate, sigma_2)
    % Compute logistic regression gradient
    n = length(y);
    l_gradient = zeros(size(w_estimate));

    for i = 1:n
        l_gradient = l_gradient + (y(i) * X(i, :)' - (exp(w_estimate' * X(i, :)') * X(i, :)') / (1 + exp(w_estimate' * X(i, :)')));
    end

    % Compute the second part of the gradient
    d_alpha_0 = computePartialAlpha(0, w_estimate, sigma_2);
    d_alpha_2 = computePartialAlpha(2, w_estimate, sigma_2);

    jeffreys_prior_gradient = 0.5 * ((length(w_estimate) - 1) * d_alpha_0 + d_alpha_2) * w_estimate / norm(w_estimate);

    gradient = l_gradient + jeffreys_prior_gradient;
end

function alpha_partial = computePartialAlpha(k, w_estimate, sigma_2)
    % Compute the partial derivative of alpha_k with respect to the norm of w_hat

    sigma = sqrt(sigma_2);

    % Generate random samples
    z = randn(size(w_estimate));

    % Compute p and expectations
    p = exp(norm(w_estimate) * sigma * z) ./ (1 + exp(norm(w_estimate) * sigma * z));
    expectation = mean(z.^(k + 1) .* (p - 3 * p.^2 + 2 * p.^3));

    % Calculate the partial derivative
    alpha_partial = sigma * expectation;
end

function cost = compute_cost(y, h)
    % Compute the logistic regression cost function
    cost = -mean(y .* log(h) + (1 - y) .* log(1 - h));
end