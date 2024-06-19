import numpy as np
from variable import Variable
from function import square, exp, add, mul
from config import no_grad


def main():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = add(mul(a, b), c)

    y.backward()

    print(f'y = {y.data}, a.grad = {a.grad}, b.grad = {b.grad}')

if __name__ == '__main__':
    main()
