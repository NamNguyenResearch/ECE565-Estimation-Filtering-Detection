clear;
clc;

% =========================================================================
% Simulation parameters:  
num_MonteCarlo = 200; % Number of Monte-Carlo runs
d = 2; % Number of features 
n = [10, 20, 50, 80, 200, 300, 500, 700, 1000]; % Number of observations/examples
w = transpose([1, 1])/sqrt(2); % Groud-truth value of w

% ML estimator parameters:
max_iters = 2000;
alpha = 0.1;
lambda_L1 = 1;

sigma_2 = [1, 15]; % Variance 

MSE_CRLB_Average = zeros(length(n),1);
MSE_Average_0 = zeros(length(n),1);
MSE_Average_1 = zeros(length(n),1);
    
for i = 1:length(n) 
    for j = 1:length(sigma_2) 
        [MSE_CRLB_Average(i,j),MSE_Average_0(i,j),MSE_Average_1(i,j)] ...
        = CRLB_function(n(i),w,d,sigma_2(j),num_MonteCarlo,max_iters,alpha,lambda_L1);
    end
end

% =========================================================================
% Plot figures:

% Plot the CRLB versus number n
figure(1); 
loglog(n,MSE_CRLB_Average(:,1),'-r',n,MSE_Average_0(:,1),'--*b',n,MSE_Average_1(:,1),'--+g','LineWidth',1)
grid on
hold on
loglog(n,MSE_CRLB_Average(:,2),'-m',n,MSE_Average_0(:,2),'--oc',n,MSE_Average_1(:,2),'--squarek','LineWidth',1)
xlabel('n');
ylabel('CRLB & MSE');
legend('CRLB, \sigma^2 = 1','MSE w/o regularization, \sigma^2 = 1','MSE w/ L1, \sigma^2 = 1',...
       'CRLB, \sigma^2 = 15','MSE w/o regularization, \sigma^2 = 15','MSE w/ L1, \sigma^2 = 15');
hold off