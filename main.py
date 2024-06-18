import numpy as np
from variable import Variable
from function import square, exp
import unittest

def main():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()

    print(f'x: {x.data}, y: {y.data}, x.grad: {x.grad}')


if __name__ == '__main__':
    main()
