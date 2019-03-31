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
m = size(X,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

centroids

for i = 1:m

  % get sample data at row
  xi = X(i,:);
  Xi = xi;

  % Duplicate data row into matrix with rows = K
  for j = 2:K
    Xi = [Xi; xi];
  end

  % Calculate the difference between xi and uj
  var = Xi - centroids;

  % Calculate the distance
  distance = var*var';

  % The distance to each centroid is on diagonal line, extract this line from distance
  diagonal = diag(distance);

  % Get the index of min value of the diagonal, this is the K to be assigned to ci
  [minval, kmin] = min(diagonal);

  idx(i) = kmin;
end



% =============================================================

end

