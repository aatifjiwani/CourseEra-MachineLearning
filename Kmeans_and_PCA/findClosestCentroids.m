function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i = 1:size(X,1)
  ex = X(i,:);
  min = intmax("uint32");  
  min_cluster = 0;
  for k = 1:K
    cluster = centroids(k,:);
    
    diff = ex - cluster;
    sq = diff.*diff;
    sum_sq = sum(sq);
    
    if (sum_sq < min)
      min = sum_sq;
      min_cluster = k;
    endif
  end
  
  idx(i) = min_cluster;
end
% =============================================================

end

