function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

errors = zeros(9);

for c_index = 1:9
  c_try = 0.01 * 3^(c_index-1);
  for sigma_index = 1:9
    sigma_try = 0.01 * 3^(sigma_index-1);
    
    model= svmTrain(X, y, c_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try)); 
    predictions = svmPredict(model, Xval);
    
    errors(c_index, sigma_index) = mean(double(predictions ~= yval))
    
  end
end

[row_min_error min_c_indexes] = min(errors);
[min_error min_sigma_index] = min(row_min_error);

min_c_index = min_c_indexes(min_sigma_index);

C = 0.01 * 3^(min_c_index-1);
sigma = 0.01 * 3^(min_sigma_index-1);

% =========================================================================

end
