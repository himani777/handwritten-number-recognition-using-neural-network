function g = sigmoidGradient(z)

g = zeros(size(z));
l=sigmoid(z);
g=l.*(1-l);

end
