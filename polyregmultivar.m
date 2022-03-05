function [err,model,errT] = polyregmultivar(x,y,lambda,xT,yT)

%% To perform multivariate Risk Regularization
%
% example: function [err,model,errT] = polyreg(x,y,lambda,xT,yT)
%
% x = vector of input scalars for training
% y = vector of output scalars for training
% D = the order plus one of the polynomial being fit
% lambda = the penalty parameter added while calculating Regulairsed Risk
% minimization
% xT = vector of input scalars for testing
% yT = vector of output scalars for testing
% err = average squared loss on training
% model = vector of polynomial parameter coefficients
% errT = average squared loss on testing

%% Model and Error Calculation for Training Dataset
[x_row,x_column] = size(x);
model = ((x'* x + eye(x_column)*lambda)^-1) * x'* y;    %With the lambda component
err   = (1/(2*x_row)) * sum((y - x*model).^2) ;
%%reg_component = (lambda/(2*x_row))*model;
%%err = Remp + reg_component;

%% Model and Error Calculation for Testing Dataset
if (nargin==5) 
  [xT_row,xT_column] = size(xT);
  xxT = xT' * xT;
  modelT = pinv(xxT + (lambda * eye(xT_column))) * xT' * yT;
  errT  = (1/(2*xT_row))*sum((yT-xT*modelT).^2);
  %%reg_componentT = (lambda/(2*xT_row))*modelT;
  %%errT = RempT + reg_componentT;
end


