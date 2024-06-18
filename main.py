import numpy as np
from variable import Variable
from function import square, exp, add


def main():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)


if __name__ == '__main__':
    main()
