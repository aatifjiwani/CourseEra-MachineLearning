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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

total = 0;
for i = 1:m
  temp_y = y(i);
  h_x = sigmoid(theta'*X(i,:)');
  total = total + (-temp_y*log(h_x) - (1-temp_y)*log(1-h_x));
end

J = (1/m)*total;

total = 0;
for i = 2:size(theta)
  total += (theta(i)*theta(i))
end

J += ((lambda/(2*m))*total);


for j = 1:size(theta)
  grad_tot = 0;
  for i = 1:m
    grad_tot += (sigmoid(theta'*X(i,:)') - y(i))*X(i,j);
  end
  grad(j) = grad_tot;
  
  if (j > 1)
    grad(j) = grad(j) + lambda*theta(j);
  end
  
end

grad = grad./m;




% =============================================================

end
