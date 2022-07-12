# Using Numpy
def relu(predictions):
    return np.maximum(0, predictions)

def single_neuron_cost(A, y, theta):
    predictions = relu(A@theta)
    cost = (np.linalg.norm(predictions, ord=2)**2)-(2*y.T@A@theta)+(np.linalg.norm(y, ord=2)**2)
    return cost

def single_neuron_grad(A, y, theta):
    predictions = relu(A@theta)
    grad = A.T@predictions-A.T@y
    return grad