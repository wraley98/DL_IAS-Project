function Y = IAS_scale(X1,X2)
%

min1 = min(X1(:));
max1 = max(X1(:));
min2 = min(X2(:));
max2 = max(X2(:));

Y = ((X2-min2)/(max2-min2))*(max1-min1) + min1;
