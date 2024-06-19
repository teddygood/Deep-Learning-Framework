import numpy as np
from autograd import Variable
from autograd.benchmark import rosenbrock

def main():
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    lr = 0.001
    iters = 1000

    for i in range(iters):
        print(x0, x1)

        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

if __name__ == '__main__':
    main()
