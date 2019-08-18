function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
cost_total = 0;

%for i = 1:num_movies
%  for j = 1:num_users
%    if (R(i,j) == 1)
%      pred = Theta(j,:)*X(i,:)';
%      cost = (pred-Y(i,j))^2;
%      cost_total = cost_total + cost;
%      
%    endif
%  end
%end

pred = (X*Theta').*R;
pred_y = Y.*R;
cost = pred-pred_y;
cost_total = sum(sum(cost.*cost));

J = cost_total / 2;

J = J + ((lambda/2)*sum(sum(Theta.*Theta))) + ((lambda/2)*sum(sum(X.*X)));

for i = 1:num_movies
  movie = X(i,:); % 1 x K
  did_predict = R(i,:);
  
  pred = (movie*Theta').*did_predict; % 1 x nU
  pred_y = (Y(i,:).*did_predict);
  
  cost = pred-pred_y; % 1 x nU
  
  X_grad(i,:) = cost*Theta + lambda.*movie;
  
  
end

for j = 1:num_users
  user = Theta(j,:); % 1 x K
  did_predict = R(:,j)'; % 1 x nM
  
  pred = (user*X').*did_predict; % 1 x nM
  pred_y = (Y(:,j)'.*did_predict);
  
  cost = pred-pred_y;
  
  Theta_grad(j,:) = cost*X + lambda.*user;
end











% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
