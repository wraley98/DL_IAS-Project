function d = IAS_kernel_dist(K1,K2)
%

[M,N] = size(K1);
Kn = zeros(M,N);

min1 = min(K1(:));
max1 = max(K1(:));
min2 = min(K2(:));
max2 = max(K2(:));

for r = 1:M
    for c = 1:N
        v1 = K1(r,c);
        Kn(r,c) = min2 + (max2-min2)*(v1-min1)/(max1-min1);
    end
end

if K2==Kn
    Kd = abs(K1-K2);
else
    Kd = abs(K2-Kn);
end
d = sum(Kd(:))/(M*N);
%d2 = max(Kd(:));
