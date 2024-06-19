import autograd.function as F


def polynomial_example(x):
    y = x ** 4 - 2 * x ** 2
    return y


def higher_order_example(x):
    y = polynomial_example(x)
    y.backward(create_graph=True)
    print(x.grad)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    print(x.grad)


def sin_higher_order_derivative(x):
    y = F.sin(x)
    y.backward(create_graph=True)

    for i in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(x.grad)


def double_backprop_ex(x):
    y = x ** 2
    y.backward(create_graph=True)
    gx = x.grad
    x.cleargrad()
    z = gx ** 3 + y
    z.backward()
    print(x.grad)
