function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    der_h = zeros(2,1); % Learning rate hypothesis
    trp = theta';
    xtrp = X';
    t0 = 0;
    t1 = 0;
    for i = 1:m
       t0 = t0 + alpha*(trp*xtrp(:,i) - y(i))/m;
       t1 = t1 + alpha*xtrp(2,i)*(trp*xtrp(:,i) - y(i))/m;
    end

    lr_theta = [t0; t1];
    theta = theta - lr_theta;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
