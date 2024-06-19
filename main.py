import numpy as np
from variable import Variable
from function import square, exp, add
from config import no_grad


def main():
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()
    print(y.grad, t.grad)  # None None
    print(x0.grad, x1.grad)  # 2.0 1.0

    with no_grad():
        x = Variable(np.array(2.0))
        y = square(x)

    print(x.data, y.data)

if __name__ == '__main__':
    main()
