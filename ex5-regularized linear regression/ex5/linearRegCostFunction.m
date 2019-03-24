function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


thetaReg = theta; % Copy theta to theta regularised
thetaReg(1) = 0; % Make first element = 0, as regularisation is not applied on first elem


hx = X*theta; % Calculate h(x) term

% Calculate linear regression cost function using vectored implementation
J = 1/(2*m) * sum((hx - y).^2);
J = J + lambda/(2*m)*sum(thetaReg.^2); % Apply regularisation to cost function


% Calculate gradient decent portion 
grad = 1/m * (X'*(hx - y));
grad = grad + (lambda/m).*thetaReg; % Apply regularisation to grad





% =========================================================================

grad = grad(:);

end
