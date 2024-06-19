import numpy as np
import autograd.function as F
from autograd import Variable
from autograd.benchmark import rosenbrock, f

def main():
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward(create_graph=True)
    print(x.grad)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    print(x.grad)

    y = F.sin(x)
    y.backward(create_graph=True)

    for i in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(x.grad)


if __name__ == '__main__':
    main()
