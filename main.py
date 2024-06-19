import numpy as np
from autograd import Variable


def main():
    x = Variable(np.array(1.0))
    y = (x + 3) ** 2
    y.backward()

    print(y)
    print(x.grad)

if __name__ == '__main__':
    main()
