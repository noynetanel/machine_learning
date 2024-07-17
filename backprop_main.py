import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *


# section b:

# Loading Data
np.random.seed(0)  # For reproducibility
n_train = 5000
n_test = 10000

# Training configuration
epochs = 30
batch_size = 10

# testing
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)
batch_size = 10
learning_rates = [0.001, 0.01, 0.1, 1, 10]
epoch = np.arange(1, 31)


train_loss = []
train_accuracies = []
test_accuracies = []

# run on all learning rates and get data
for l_rate in learning_rates:
    print(l_rate)
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)
    parameters, train_loss_epoch, test_loss_epoch, train_accu_epoch, test_accu_epoch = net.train(x_train, y_train, epochs, batch_size, l_rate, x_test=x_test, y_test=y_test)
    train_accuracies.append(train_accu_epoch)
    test_accuracies.append(test_accu_epoch)
    train_loss.append(train_loss_epoch)

# now plot the required graphs
# training accuracy graph
for i in range(len(learning_rates)):
    plt.plot(epoch, train_accuracies[i], label="rate "+str(learning_rates[i]))
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.title('Training accuracy')
plt.legend()
plt.grid(True)
plt.show()

# training loss graph
for i in range(len(learning_rates)):
    plt.plot(epoch, train_loss[i], label="rate " + str(learning_rates[i]))
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.title('Training loss')
plt.legend()
plt.grid(True)
plt.show()

# testing accuracy graph
for i in range(len(learning_rates)):
    plt.plot(epoch, test_accuracies[i], label="rate: "+str(learning_rates[i]))
plt.xlabel('epochs')
plt.ylabel('testing accuracy')
plt.title('Testing accuracy')
plt.legend()
plt.grid(True)
plt.show()

#  section c:

# we need to train on the entire dataset
learning_rate = 0.1  # Assuming this is the optimal learning rate from previous experiments
np.random.seed(0)
n_train = 50000
n_test = 10000
batch_size = 10
epochs = 30
# Network configuration
layer_dims = [784, 40, 10]
net = Network(layer_dims)
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

# train the network
parameters, train_loss_epoch, test_loss_epoch, train_accu_epoch, test_accu_epoch = net.train(
    x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

# Get the final test accuracy
final_epoch_test_acc = test_accu_epoch[-1]
print(f"final epoch test accuracy: {final_epoch_test_acc:}")

# section d:

# Load data
np.random.seed(0)
n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

# training configuration
epochs = 30
batch_size = 10
learning_rate = 0.1

# network configuration
layer_dims = [784, 10]
epoch = np.arange(1, 31)
net = Network(layer_dims)
parameters, train_loss_epoch, test_loss_epoch, train_accu_epoch, test_accu_epoch = net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

# graph training accuracy
plt.plot(epoch, train_accu_epoch, label="rate: 0.1")
plt.title('Training accuracy')
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.legend()
plt.grid(True)
plt.show()

# graph testing accuracy
plt.plot(epoch, test_accu_epoch, label="rate: 0.1")
plt.title('Testing accuracy')
plt.xlabel('epochs')
plt.ylabel('testing accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Display the weights
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, a in enumerate(axes.flat):
    curr_W = parameters['W1'][i, :]
    a.imshow(np.reshape(curr_W, (28, 28)), cmap='viridis', interpolation="nearest")
    a.set_title(f'class {i}')
    a.axis('off')
plt.colorbar(axes[0, 0].images[0], ax=axes, orientation='vertical', fraction=.1)
plt.show()


# section e:

# we need to train on the entire dataset
learning_rate = 0.1  # Assuming this is the optimal learning rate from previous experiments
np.random.seed(0)
n_train = 50000
n_test = 10000
batch_size = 140
epochs = 30
# Network configuration
layer_dims = [784, 40, 10]
net = Network(layer_dims)
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

# train the network
parameters, train_loss_epoch, test_loss_epoch, train_accu_epoch, test_accu_epoch = net.train(
    x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

# Get the final test accuracy
final_epoch_test_acc = test_accu_epoch[-1]
print(f"final epoch test accuracy: {final_epoch_test_acc:}")
