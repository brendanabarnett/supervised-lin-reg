import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt     # to graph predictions


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=input_size, out_features=output_size)     #linear model

    def forward(self, x):   #takes next step
        out = self.linear(x)
        return out
    

def linRegTrain(x_train_vals, y_train_vals, criterion, optimizer, epochs=100):
    #reshape to 1 column
    x_train = np.array(x_train_vals, dtype=np.float32).reshape(-1, 1)
    y_train = np.array(y_train_vals, dtype=np.float32).reshape(-1,1)

    #convert to pytorch variable
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    for epoch in range(epochs):
        optimizer.zero_grad()               # clears gradient from previous iteration
        outputs = model(inputs)             # calls linearRegression
        loss = criterion(outputs, labels)   # calculates mean squared error
        loss.backward()                     # computes gradient loss for each parameter
        optimizer.step()                    # updates each parameter using grad

        print(f'epoch {epoch+1}, loss {loss.item()}')     # prints current iteration and loss


def linRegTest(x_test):
    with torch.no_grad():   # clears gradient
        predicted = model(Variable(torch.from_numpy(x_test))).data.numpy()      # make prediction for test data
        print("\nPredicted:\n", predicted)            # print predicted y vals
    return predicted


def linRegPlotResults(x_test, y_test, predicted):
    # displays test data and prediction on a graph
    plt.clf()
    plt.plot(x_test, y_test, 'go', label='Test Data', alpha=0.5)
    plt.plot(x_test, predicted, '--', label='Prediction', alpha=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # feel free to edit the info below to further understand what each field does! 

    # training data: y = 2x + 1
    x_train_vals = [i for i in range(15)]
    y_train_vals = [i * 2 + 1 for i in x_train_vals]

    # testing data
    x_test_vals = [i + 20 for i in range(15)]
    y_test_vals = [i * 2 + 1 for i in x_test_vals]

    inputDim = 1
    outputDim = 1
    alpha = 0.01        # learning rate
    epochs = 250
    model = LinearRegression(inputDim, outputDim)       # lin reg model (defined above)
    criterion = torch.nn.MSELoss()                                      # loss function: mean squared error (minimize this)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=alpha)    # optimization function: Stochastic Gradient Descent (tool we use to minimize MSE)

    linRegTrain(x_train_vals, y_train_vals, criterion, optimizer, epochs)

    x_test = np.array(x_test_vals, dtype=np.float32).reshape(-1, 1)
    y_test = np.array(y_test_vals, dtype=np.float32).reshape(-1, 1)

    predicted = linRegTest(x_test)
    linRegPlotResults(x_test, y_test, predicted)
