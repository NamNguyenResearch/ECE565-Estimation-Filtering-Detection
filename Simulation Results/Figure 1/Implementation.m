clear;
clc;

% Simulation parameters:
num_MonteCarlo = 200;   % Number of Monte-Carlo runs
d = 2;                  % Number of features
sigma_2 = 1;            % Variance
n = [10, 20, 50, 80, 200, 300, 500, 700, 1000];   % Number of observations/examples
w = [1; 1] / sqrt(2);    % Ground-truth value of w

% ML estimator parameters:
max_iters = 2000;
alpha = 0.1;
lambda_L1 = 1;
lambda_L2 = 1;

MSE_CRLB_Average = zeros(length(n), 1);
MSE_Average_0 = zeros(length(n), 1);
cost_function_history_Average_0 = zeros(max_iters, length(n));
MSE_Average_1 = zeros(length(n), 1);
cost_function_history_Average_1 = zeros(max_iters, length(n));
MSE_Average_2 = zeros(length(n), 1);
cost_function_history_Average_2 = zeros(max_iters, length(n));
MSE_Average_3 = zeros(length(n), 1);
cost_function_history_Average_3 = zeros(max_iters, length(n));
MSE_Average_4 = zeros(length(n), 1);
cost_function_history_Average_4 = zeros(max_iters, length(n));

for i = 1:length(n)
    [MSE_CRLB_Average(i), cost_function_history_Average_0(:, i), MSE_Average_0(i),...
     cost_function_history_Average_1(:, i), MSE_Average_1(i), cost_function_history_Average_2(:, i), MSE_Average_2(i),...
     cost_function_history_Average_3(:, i), MSE_Average_3(i),cost_function_history_Average_4(:, i), MSE_Average_4(i)] = CRLB_function(n(i), w, d, sigma_2, num_MonteCarlo, max_iters, alpha,lambda_L1,lambda_L2);
end

% Plot figures:

% Plot the CRLB versus number n
figure(1);
loglog(n, MSE_CRLB_Average, '-r', n, MSE_Average_0, '--squareb',n, MSE_Average_1, '--og',n, MSE_Average_2, '--*m',n, MSE_Average_3,'-+k',n, MSE_Average_4,'--diamondc');
grid on;
xlabel('n');
ylabel('CRLB & MSE');
legend('CRLB','MSE w/o regularization','MSE w/ L_{1}', 'MSE w/ L_{2}','Iterative Scaling Ver1','Iterative Scaling Ver2');

% % Plot the log-likelihood versus number of iterations
% figure(2);
% plot(1:max_iters, mean(cost_function_history_Average_4, 2), '-b', 'LineWidth', 1);
% xlabel('Number of Iterations');
% ylabel('Cost function');
% grid on;
% legend('Jeffreys prior');