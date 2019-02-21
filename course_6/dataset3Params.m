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

c_and_s_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

predictions_error = zeros(length(c_and_s_vals),length(c_and_s_vals));

for i=1:length(c_and_s_vals) 
    for j=1:length(c_and_s_vals) 
        c = c_and_s_vals(i);
        s = c_and_s_vals(j);
        model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
        predictions = svmPredict(model, Xval);
        predictions_error(i,j) = mean(double(predictions ~= yval));
    end
end

minimum = min(min(predictions_error));
[i, j] = find(predictions_error == minimum);
C = c_and_s_vals(i);
sigma = c_and_s_vals(j);

% =========================================================================

end
