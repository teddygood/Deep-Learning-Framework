import numpy as np
from autograd import Variable
from autograd.example import higher_order_example, double_backprop_ex, sin_higher_order_derivative


def main():

    # higher order example
    x = Variable(np.array(2.0))
    higher_order_example(x)

    # sin higher order derivative
    x = Variable(np.array(2.0))
    sin_higher_order_derivative(x)

    # double backpropagation
    x = Variable(np.array(2.0))
    double_backprop_ex(x)


if __name__ == '__main__':
    main()
