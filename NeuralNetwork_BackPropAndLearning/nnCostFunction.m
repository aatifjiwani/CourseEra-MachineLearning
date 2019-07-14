function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Feeding forward

X = [ones(m, 1) X];
z_2 = X * Theta1'; 
a_2 = sigmoid(z_2); % 5000 x 25

a_2 = [ones(size(a_2, 1), 1) a_2];
z_3 = a_2 * Theta2';

h_x = sigmoid(z_3); % 5000 x 10

% Computing the cost

y_matrix = eye(num_labels)(y,:);

total_m = zeros(m ,1);

for i = 1:num_labels
  y_double = y_matrix(:,i);
  h_double = h_x(:,i);
  
  temp_total = (-y_double.*log(h_double) - (1.-y_double).*log(1.-h_double));
  total_m += temp_total;
end

J = (1/m)*sum(total_m);

theta1_v = Theta1(:,2:end)(:);
theta1_sum = sum(theta1_v.*theta1_v);
theta2_v = Theta2(:,2:end)(:);
theta2_sum = sum(theta2_v.*theta2_v);

J = J + (lambda/(2*m))*(theta1_sum + theta2_sum);

%computing the gradient for each layer of Theta

D2 = zeros(num_labels, size(a_2, 2)); %10 x 26
D1 = zeros(size(z_2, 2), size(X, 2)); % 25 x 401

for i = 1:m 
  delta_3 =  (h_x(i,:) - y_matrix(i,:))'; %10 x 1
  delta_2 = Theta2(:,2:end)'*delta_3.*sigmoidGradient(z_2(i,:)'); %25 x 1
  D2 = D2 + delta_3*a_2(i,:);
  D1 = D1 + delta_2*X(i,:);
end

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = (1/m).*D1 + (lambda/m).*Theta1;
Theta2_grad = (1/m).*D2 + (lambda/m).*Theta2;











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
