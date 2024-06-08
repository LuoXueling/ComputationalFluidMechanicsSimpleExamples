J1 = -2;
J2 = 2;
N = J2 - J1 + 1;
n = 2;

A = zeros(N,N);
%i为泰勒展开第i项
%j为求和变量
for i = 1:N
    for j = 1:N
        A(i,j) = (j-1+J1)^(i-1)/factorial(i-1);
    end
end

B = zeros(N,1);
for i = 1:N
    B(i) = (i == n) * 1;
end

X = A\B;

i = N + 1;
tmp = 0;
acc = -n;
while 1
    for j = 1:N
        tmp = tmp + X(j)*(j-1+J1)^(i-1)/factorial(i-1);
    end
    if abs(tmp) > 1e-5
        acc = i - 1 - n;
        break;
    end
    if i > 1000
        acc = -1;
        break;
    end
    i = i + 1;
end