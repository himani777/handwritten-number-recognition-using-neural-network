function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels, X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

[m, n] = size(X);
X=[ones(m,1) X];
t11=Theta1*(X');
t1=sigmoid(t11);
t2=[ones(1,m);t1];
temp=Theta2*t2;

h=sigmoid(temp);
yp=[];
nm=size(h,1);
for i=1:m
 tempy=zeros(nm,1);
 tempy(y(i))=1;
 yp=[yp tempy];
end
term1=yp.*log(h);
term2=(1-yp).*log(1-h);

J=(-1/m)*(sum(sum(term1))+sum(sum(term2)));

sqt1=Theta1.^2;
sqt2=Theta2.^2;
sqt1(:,1)=zeros(size(sqt1,1),1);
sqt2(:,1)=zeros(size(sqt2,1),1);

J=J+(lambda*(sum(sum(sqt1))+sum(sum(sqt2)))/(2*m));


for t=1:m
 a1=X(t,:)';
 a2=[1; t1(:,t)];
 z2=t11(:,t);
 a3=h(:,t);
 yi=yp(:,t);
 d3=a3-yi;
 d2=(Theta2')*d3.*sigmoidGradient([1;z2]);
 d2=d2(2:end);
 
 Theta1_grad=Theta1_grad+(d2*(a1'));
 Theta2_grad=Theta2_grad+(d3*(a2'));
 

end

k1=(lambda/m)*Theta1;
k1(:,1)=zeros(size(k1,1),1);
k2=(lambda/m)*Theta2;
k2(:,1)=zeros(size(k2,1),1);

Theta1_grad=(1/m)*Theta1_grad+k1;
Theta2_grad=(1/m)*Theta2_grad+k2;


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
