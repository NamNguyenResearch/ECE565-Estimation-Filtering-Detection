clear;
clc;

% =========================================================================
% Simulation parameters:  
num_MonteCarlo = 400; % Number of Monte-Carlo runs
d = 2; % Number of features 
sigma_2 = 1; % Variance 
n = [50, 100, 1000]; % Number of observations/examples
w = transpose([1, 1])/sqrt(2); % Groud-truth value of w

norm_w = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20];

for i = 1:length(norm_w) 
    w_range(:,i) = norm_w(i)*w;
end

% ML estimator parameters:
max_iters = 2500;
alpha = 0.1;

for i = 1:length(w_range)  
    for j = 1:length(n) 
        [MSE_CRLB_Average(i,j),cost_function_history_Average_0(:,i,j),MSE_Average_0(i,j)] ...
        = CRLB_function(n(j),w_range(:,i),d,sigma_2,num_MonteCarlo,max_iters,alpha);
    end
end 


% =========================================================================
% Plot figures:

% Plot the CRLB versus norm(w)
figure(1); 
loglog(norm_w,MSE_CRLB_Average(:,1),'-r',norm_w,MSE_Average_0(:,1),'--*b','LineWidth',1)
grid on
hold on
loglog(norm_w,MSE_CRLB_Average(:,2),'-m',norm_w,MSE_Average_0(:,2),'--og','LineWidth',1)
loglog(norm_w,MSE_CRLB_Average(:,3),'-c',norm_w,MSE_Average_0(:,3),'--squarek','LineWidth',1)
xlabel('||w||');
ylabel('CRLB & MSE');
legend('CRLB (n=50)','MSE (n=50)','CRLB (n=100)','MSE (n=100)','CRLB (n=1000)','MSE (n=1000)');
hold off


% % Plot the log-likelihood versus number of iterations
% figure(2);
% plot(1:length(cost_function_history_Average_0),cost_function_history_Average_0(:,6), '-b','LineWidth',1);
% xlabel('Number of Iterations');
% ylabel('Cost function');
% grid on;
% legend('No regularization');