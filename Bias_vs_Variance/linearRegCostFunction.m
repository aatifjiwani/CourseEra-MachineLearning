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

%computing cost
h_x = X*theta;

error = h_x - y;
error = error.*error;

cost = (1/(2*m))*(sum(error));

theta_reg = theta(2:end);
sum_reg = sum(theta_reg.*theta_reg);

J = cost + (lambda/(2*m))*sum_reg

%computing gradient
error = h_x - y;
grad_error = (1/m).*(X'*error);

theta_reg = [0; theta(:)(2:end)];

grad = grad_error + (lambda/m).*theta_reg;







% =========================================================================

%grad = grad(:);

end
