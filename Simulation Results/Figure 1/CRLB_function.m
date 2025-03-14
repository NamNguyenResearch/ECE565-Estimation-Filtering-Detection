function [MSE_CRLB_Average,cost_function_history_Average_0,MSE_Average_0,cost_function_history_Average_1,MSE_Average_1,...
          cost_function_history_Average_2,MSE_Average_2,cost_function_history_Average_3,MSE_Average_3,...
          cost_function_history_Average_4,MSE_Average_4] = CRLB_function(n,w,d,sigma_2,num_MonteCarlo,max_iters,alpha,lambda_L1,lambda_L2)

    % Simulation parameters:
    mu = zeros(d, 1);          % Mean vector
    Sigma = sigma_2 * eye(d);   % Covariance matrix

    MSE_CRLB = zeros(num_MonteCarlo, 1);
    MSE_0 = zeros(num_MonteCarlo, 1);
    MSE_1 = zeros(num_MonteCarlo, 1);
    MSE_2 = zeros(num_MonteCarlo, 1);
    MSE_3 = zeros(num_MonteCarlo, 1);
    MSE_4 = zeros(num_MonteCarlo, 1);

    cost_function_history_matrix_0 = zeros(max_iters, num_MonteCarlo);
    cost_function_history_matrix_1 = zeros(max_iters, num_MonteCarlo);
    cost_function_history_matrix_2 = zeros(max_iters, num_MonteCarlo);
    cost_function_history_matrix_3 = zeros(max_iters, num_MonteCarlo);
    cost_function_history_matrix_4 = zeros(max_iters, num_MonteCarlo);

    for i = 1:num_MonteCarlo
        % Generate n feature vectors
        X = mvnrnd(mu, Sigma, n);

        % Calculate corresponding labels using vectorized approach
        p_y_1 = exp(X * w) ./ (1 + exp(X * w));
        y = rand(n, 1) < p_y_1;

        % Calculate CRLB
        u_1 = w / norm(w);
        a = sqrt(sigma_2) * norm(w);
        CRLB = (n * sigma_2 * alpha_function(0, a))^(-1) * (eye(d) - (alpha_function(2, a) - alpha_function(0, a)) / alpha_function(2, a) * (u_1 * u_1'));

        % CRLB of MSE
        MSE_CRLB(i) = trace(CRLB);

        % Calculate MSE of ML estimators using IterativeScaling2
        [w_estimate_0, cost_function_history_matrix_0(:,i)] = NoRegularization(X,y,alpha,max_iters);
        [w_estimate_1, cost_function_history_matrix_1(:,i)] = L1Regularization(X,y,alpha,lambda_L1,max_iters);
        [w_estimate_2, cost_function_history_matrix_2(:,i)] = L2Regularization(X,y,alpha,lambda_L2,max_iters);
        [w_estimate_3, cost_function_history_matrix_3(:,i)] = Iterative_Scaling1(X,y,max_iters,d);
        [w_estimate_4, cost_function_history_matrix_4(:,i)] = Iterative_Scaling2(X,y,max_iters,d);

        MSE_0(i) = norm(w_estimate_0 - w)^2;
        MSE_1(i) = norm(w_estimate_1 - w)^2;
        MSE_2(i) = norm(w_estimate_2 - w)^2;
        MSE_3(i) = norm(w_estimate_3 - w)^2;
        MSE_4(i) = norm(w_estimate_4 - w)^2;
    end

    % Calculate the average value of CRLB
    MSE_CRLB_Average = mean(MSE_CRLB);

    % Calculate the average value of MSE of ML estimators
    MSE_Average_0 = mean(MSE_0);
    MSE_Average_1 = mean(MSE_1);
    MSE_Average_2 = mean(MSE_2);
    MSE_Average_3 = mean(MSE_3);
    MSE_Average_4 = mean(MSE_4);

    % Calculate the average cost function history
    cost_function_history_Average_0 = mean(cost_function_history_matrix_0, 2);
    cost_function_history_Average_1 = mean(cost_function_history_matrix_1, 2);
    cost_function_history_Average_2 = mean(cost_function_history_matrix_2, 2);
    cost_function_history_Average_3 = mean(cost_function_history_matrix_3, 2);
    cost_function_history_Average_4 = mean(cost_function_history_matrix_4, 2);
end

function [result] = alpha_function(k,a)
    fun = @(z) 1./sqrt(2.*pi).*exp(a.*z-1./2.*z.^2).*(z.^k)./((1+exp(a.*z)).^2);
    result = integral(fun, -Inf, Inf);
end