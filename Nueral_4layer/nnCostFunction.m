function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels, X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)),hidden_layer_size1, (input_layer_size + 1));
temp=hidden_layer_size1 * (input_layer_size + 1);
Theta2 = reshape(nn_params((1 + temp):(temp+(hidden_layer_size2*(hidden_layer_size1 + 1)))),hidden_layer_size2, (hidden_layer_size1 + 1));
Theta3 = reshape(nn_params((temp+(hidden_layer_size2*(hidden_layer_size1 + 1))+1):end),num_labels, (hidden_layer_size2 + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

[m, n] = size(X);

t11=[ones(m,1) X];
temp1=Theta1*(t11');
h1=sigmoid(temp1);

t22=[ones(1,m);h1];
temp2=Theta2*t22;
h2=sigmoid(temp2);

t33=[ones(1,m);h2];
temp3=Theta3*t33;
h3=sigmoid(temp3);


yp=[];

nm=size(h3,1);

for i=1:m
 tempy=zeros(nm,1);
 tempy(y(i))=1;
 yp=[yp tempy];
end


term1=yp.*log(h3);
term2=(1-yp).*log(1-h3);

J=(-1/m)*(sum(sum(term1))+sum(sum(term2)));

sqt1=Theta1.^2;
sqt2=Theta2.^2;
sqt3=Theta3.^2;
sqt1(:,1)=zeros(size(sqt1,1),1);
sqt2(:,1)=zeros(size(sqt2,1),1);
sqt3(:,1)=zeros(size(sqt3,1),1);

J=J+(lambda*(sum(sum(sqt1))+sum(sum(sqt2)) + sum(sum(sqt3)))/(2*m));


for t=1:m
 a1=X(t,:)';
 a1=[1; a1];
 a2=[1; h1(:,t)];
 z2=temp1(:,t);
 
 a3=[1;h2(:,t)];
 z3=temp2(:,t);
 
 a4=h3(:,t);
 
 yi=yp(:,t);
 
 d4=a4-yi;
 d3=(Theta3')*d4.*sigmoidGradient([1;z3]);
 d3=d3(2:end);
 d2=(Theta2')*d3.*sigmoidGradient([1;z2]);

 d2=d2(2:end);
 
 Theta1_grad=Theta1_grad+(d2*(a1'));
 Theta2_grad=Theta2_grad+(d3*(a2')); 
 Theta3_grad=Theta3_grad+(d4*(a3'));
 

end

k1=(lambda/m)*Theta1;
k1(:,1)=zeros(size(k1,1),1);
k2=(lambda/m)*Theta2;
k2(:,1)=zeros(size(k2,1),1);
k3=(lambda/m)*Theta3;
k3(:,1)=zeros(size(k3,1),1);

Theta1_grad=(1/m)*Theta1_grad+k1;
Theta2_grad=(1/m)*Theta2_grad+k2;
Theta3_grad=(1/m)*Theta3_grad+k3;

grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
