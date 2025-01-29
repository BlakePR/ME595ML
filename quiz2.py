import sympy as sp

sigmoid = lambda x: 1 / (1 + sp.exp(-x))
dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))

p2 = dsigmoid(0)

print(p2)

x = 3; w = 2; b = 1
z = w * x + b
y = 0.8
yout = sigmoid(z)

dL_dyout = 2 * (yout - y)
dyout_dz = dsigmoid(z)
dz_dw = x

dL_dw = dL_dyout * dyout_dz * dz_dw
print(dL_dw.evalf())