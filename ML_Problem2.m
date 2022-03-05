%% Performing cross validation on the dataset to minimize regularised risk 
%%
%%
%% Clear and Load Dataset
clear
load ('problem2.mat' )


%% Initialization and Dataset Split
 
Var_Length = length(y);

[m,n] = size(x);
P = 0.60 ;                                  %Percentage to divide the train and test data upon

training_x = x(1:P*m,:);                    %Sequential division of train data for x
training_y = y(1:P*Var_Length);             %Sequential division of train data for y

testing_x = x((P*m)+1:end,:);               %Sequential division of test data for x
testing_y = y((P*Var_Length)+1:end);        %Sequential division of test data for y
    


%% Empty array for the Regularised risk error values to be stored based on lambda values
training_errors = [];            
testing_errors = [];           



%% Regularised Risk Calculation
lambda_set = 1:1000;

for lambda = lambda_set                             %Iterate for lambda from 1 to 1000 to find the Rreg of all points at respective lambda
    
    [err, model, errT] = polyregmultivar(training_x, training_y, lambda, testing_x, testing_y); 
    
    training_errors =[training_errors, err];        % Store the train errors into an array 
    testing_errors = [testing_errors, errT];        % Store the test errors into an array 
end



%% Plotting 1/Lambda vs Train and Test Errors
figure("Name","CrossValidation of Polyvariate Function ");

clf ; 
plot(1./lambda_set, testing_errors , 'g')                   %Plot the Remp for test dataset wtr to 1/lambda value
hold on ;
plot(1./lambda_set, training_errors , 'b')                  %Plot the Remp for train dataset wtr to 1/lambda value
[min_val, minimum] = min(testing_errors);                   %This calculates the minimum value of Remp from the test dataset
plot(minimum , testing_errors(minimum), 'rx');              %Plot point where you find the least error in Remp of test dataset 

fprintf('The regularised risk is minimum at:  %d', minimum);

%% Graph Styling
xline(minimum, '--r');                                      %This line indicate the exact place where we found minimum error in test dataset
xlabel('1/Lambda');                                         % X Axis label
ylabel('Error');                                            % Y Axis label
legend('Test', 'Train');                                    % Legend to identify the lines belong to which dataset
title("Regularized Risk Lambda vs Error")                   %Title of the graph
