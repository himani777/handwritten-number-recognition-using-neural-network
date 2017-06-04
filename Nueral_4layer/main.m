clear ; close all; clc
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size1 = 25;   % 25 hidden units layer 1
hidden_layer_size2 = 30;  % 30 hidden units layer 2
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
fprintf('Loading and Visualizing Data ...\n')

load('datainput.mat');
sel=randperm(size(X,1));
X=X(sel,:);
y=y(sel,:);
T=X(4901:end,:);
X=X(1:4900,:);
Y=y(4901:end,:);
y=y(1:4900,:);
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size1);
initial_Theta2 = randInitializeWeights(hidden_layer_size1, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2 , num_labels);


initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

fprintf('\n Training Neural Network... \n')

options = optimset('MaxIter', 80);
lambda = 1;

costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size1,hidden_layer_size2,num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)),hidden_layer_size1, (input_layer_size + 1));
temp=hidden_layer_size1 * (input_layer_size + 1);
Theta2 = reshape(nn_params((1 + temp):(temp+(hidden_layer_size2*(hidden_layer_size1 + 1)))),hidden_layer_size2, (hidden_layer_size1 + 1));
Theta3 = reshape(nn_params((temp+(hidden_layer_size2*(hidden_layer_size1 + 1))+1):end),num_labels, (hidden_layer_size2 + 1));


pred = predict(Theta1, Theta2,Theta3,T);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);
sel = sel(1:20);
displayData(X(sel, :));
sol= predict(Theta1,Theta2,Theta3, X(sel,:));
fprintf('\nSoltuion to 20 examples are \n');
sol

