function [MSE_CRLB_Average,cost_function_history_Average_0,MSE_Average_0] = CRLB_function(n,w,d,sigma_2,num_MonteCarlo,max_iters,alpha) 
 
    % =====================================================================
    % Simulation parameters:   
    mu = 0*eye(d,1); % Mean vector
    Sigma = sigma_2*eye(d); % Corvariance matrix

    MSE_CRLB = zeros(num_MonteCarlo,1);

    for i = 1:num_MonteCarlo 
        % =================================================================
        % I. Calculate CRLB:

        % Generate n feature vectors:
        X = mvnrnd(mu,Sigma,n);
        y = zeros(n,1); % Label vector
        
        % =================================================================
        % Calculate their corresponding label:
        for j = 1:n 
            p_y_1 = exp(transpose(w)*transpose(X(j,:)))/(1 + exp(transpose(w)*transpose(X(j,:))));
            y(j) = rand() < p_y_1;
        end
        
        % =================================================================
        % Calculate CRLB:
        u_1 = w/norm(w);
        a = sqrt(sigma_2)*norm(w);
        
        CRLB = (n*sigma_2*alpha_function(0,a))^(-1)*(eye(d) - (alpha_function(2,a)-alpha_function(0,a))/alpha_function(2,a)*u_1*transpose(u_1));
        
        % CRLB of MSE:
        MSE_CRLB(i) = trace(CRLB);
        
        % =================================================================
        % II. Calculate MSE of ML estimators:
        
        % No regularization
        [w_estimate_0,cost_function_history_matrix_0(:,i)] = NoRegularization(X,y,alpha,max_iters);

        MSE_0(i) = (norm(w_estimate_0 - w))^2; 
    end
    
    % Calculate the average value of CRLB:
    MSE_CRLB_Average = sum(MSE_CRLB)/num_MonteCarlo;
    
    % Calculate the average value of MSE of ML estimators 
    MSE_Average_0 = sum(MSE_0)/num_MonteCarlo;

    cost_function_history_Average_0 = zeros(max_iters,1);

    for i = 1:num_MonteCarlo 
        cost_function_history_Average_0 = cost_function_history_Average_0 + cost_function_history_matrix_0(:,i);
    end

    cost_function_history_Average_0 = cost_function_history_Average_0/num_MonteCarlo;
end

function [result] = alpha_function(k,a) 
    fun = @(z) 1./sqrt(2.*pi).*exp(a.*z-1./2.*z.^2).*(z.^k)./((1+exp(a.*z)).^2);
    result = integral(fun,-Inf,Inf);
end