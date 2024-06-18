import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        # data, grad is ndarray
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # Implement with recursive
        # f = self.creator # 1. Get function
        # if f is not None:
        #     x = f.input # 2. Get input of function
        #     x.grad = f.backward(self.grad) # 3. Call function's backward method
        #     x.backward() # 4. Recursive call

        # Implement with while loop
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # Get function
            x, y = f.input, f.output  # Get input, output of function
            x.grad = f.backward(y.grad)  # Call function's backward method

            if x.creator is not None:
                funcs.append(x.creator)  # Add previous function to list


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
