function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sigx_vector = X*theta; % Calculate XO
sig = sigmoid(sigx_vector); % Calculate g(XO) based on sigmoid function

% Calculate cost function using vectored implementation
J = 1/m * (-y'*log(sig) - (1 - y)'*log(1 - sig));
for i = 2: n
  J = J + lambda/(2*m)*theta(i)^2;
end

% Calculate gradient decent portion 
regularised = ones(n);
regularised(1) = 0;
grad = 1/m * (X'*(sig - y));

for j = 2: n
  grad(j) = grad(j) + lambda/m*theta(j);
end



% =============================================================

end
