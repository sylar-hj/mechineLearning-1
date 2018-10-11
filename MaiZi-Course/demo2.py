import mnist_loader, network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('data/mnist.pkl.gz')
net = network2.Network([784, 100, 50, 25, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
net.default_weight_initializer()
print("Start training.")
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
net.SGD(training_data, epochs=30000, mini_batch_size=10, eta=0.1,
        evaluation_data=validation_data, lmbda=1.0,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True,
        monitor_evaluation_cost=True)
