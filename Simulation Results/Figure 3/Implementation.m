clear;
clc;

% =========================================================================
% Simulation parameters:  
num_MonteCarlo = 200; % Number of Monte-Carlo runs
d = 2; % Number of features 
sigma_2 = 1; % Variance 
n = [50, 100, 1000]; % Number of observations/examples
w = transpose([1, 1])/sqrt(2); % Groud-truth value of w

norm_w = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20];

for i = 1:length(norm_w) 
    w_range(:,i) = norm_w(i)*w;
end

% ML estimator parameters:
max_iters = 2000;
alpha = 0.1;
lambda_L1 = 0.1;
lambda_L2 = 0.1;

MSE_CRLB_Average = zeros(length(norm_w),length(n));
MSE_Average_0 = zeros(length(norm_w),length(n));

for i = 1:length(w_range) 
    for j = 1:length(n) 
        [MSE_CRLB_Average(i,j),cost_function_history_Average_0(:,i,j),MSE_Average_0(i,j), ...
        cost_function_history_Average_1(:,i,j),MSE_Average_1(i,j),cost_function_history_Average_2(:,i,j),MSE_Average_2(i,j)] ...
        = CRLB_function(n(j),w_range(:,i),d,sigma_2,num_MonteCarlo,max_iters,alpha,lambda_L1,lambda_L2);
    end
end 

% =========================================================================
% Plot figures:

% Plot the CRLB versus number n
figure(1); 
loglog(norm_w,MSE_CRLB_Average(:,1),'-r',norm_w,MSE_Average_0(:,1),'--*b',norm_w,MSE_Average_1(:,1),'--+g', ...
       norm_w,MSE_Average_2(:,1),'--om','LineWidth',1)
grid on
xlabel('n');
ylabel('CRLB & MSE');
legend('CRLB','MSE w/o regularization','MSE w/ L1','MSE w/ L2');

figure(2);
loglog(norm_w,MSE_CRLB_Average(:,2),'-r',norm_w,MSE_Average_0(:,2),'--*b',norm_w,MSE_Average_1(:,2),'--+g', ...
       norm_w,MSE_Average_2(:,2),'--om','LineWidth',1)
grid on
xlabel('n');
ylabel('CRLB & MSE');
legend('CRLB','MSE w/o regularization','MSE w/ L1','MSE w/ L2');

figure(3);
loglog(norm_w,MSE_CRLB_Average(:,3),'-r',norm_w,MSE_Average_0(:,3),'--*b',norm_w,MSE_Average_1(:,3),'--+g', ...
       norm_w,MSE_Average_2(:,3),'--om','LineWidth',1)
grid on
xlabel('n');
ylabel('CRLB & MSE');
legend('CRLB','MSE w/o regularization','MSE w/ L1','MSE w/ L2');


% % Plot the log-likelihood versus number of iterations
% figure(2);
% plot(1:length(cost_function_history_Average_0),cost_function_history_Average_0(:,6), '-b',...
%      1:length(cost_function_history_Average_1),cost_function_history_Average_1(:,6), '-g',...
%      1:length(cost_function_history_Average_2),cost_function_history_Average_2(:,6), '-r','LineWidth',1);
% xlabel('Number of Iterations');
% ylabel('Cost function');
% grid on;
% legend('No regularization','L1','L2');