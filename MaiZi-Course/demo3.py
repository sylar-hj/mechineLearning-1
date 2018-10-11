import matplotlib.pyplot as plt

import mnist_loader, network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net1 = network2.Network([784, 30, 10], cost=network2.QuadraticCost,
                        activate_function_and_derivation=(network2.sigmoid, network2.sigmoid_deriv),
                        SoftmaxLayerFlag=False)

net2 = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost,
                        activate_function_and_derivation=(network2.sigmoid, network2.sigmoid_deriv),
                        SoftmaxLayerFlag=False
                        )

net3 = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost,
                        activate_function_and_derivation=(network2.ReLU, network2.ReLU_deriv),
                        SoftmaxLayerFlag=False
                        )
net4 = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost,
                        activate_function_and_derivation=(network2.ReLU, network2.ReLU_deriv),
                        SoftmaxLayerFlag=False
                        )
net5 = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost,
                        activate_function_and_derivation=(network2.ReLU, network2.ReLU_deriv),
                        SoftmaxLayerFlag=True
                        )

net6 = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost,
                        activate_function_and_derivation=(network2.ReLU, network2.ReLU_deriv),
                        SoftmaxLayerFlag=False
                        )
# net.large_weight_initializer()
net1.default_weight_initializer()
print("Start training.")
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
evaluation_cost1, evaluation_accuracy1, \
training_cost1, training_accuracy1 = \
    net1.SGD(training_data, epochs=50, mini_batch_size=10, eta=0.01,
             evaluation_data=validation_data, lmbda=1.0,
             monitor_evaluation_accuracy=True)

evaluation_cost2, evaluation_accuracy2, \
training_cost2, training_accuracy2 = \
    net2.SGD(training_data, epochs=50, mini_batch_size=10, eta=0.01,
             evaluation_data=validation_data, lmbda=1.0,
             monitor_evaluation_accuracy=True)
evaluation_cost3, evaluation_accuracy3, \
training_cost3, training_accuracy3 = \
    net3.SGD(training_data, epochs=50, mini_batch_size=10, eta=0.01,
             evaluation_data=validation_data, lmbda=1.0,
             monitor_evaluation_accuracy=True)
evaluation_cost4, evaluation_accuracy4, \
training_cost4, training_accuracy4 = \
    net4.SGD(training_data, epochs=50, mini_batch_size=10, eta=0.01,
             evaluation_data=validation_data, lmbda=1.0,
             monitor_evaluation_accuracy=True)
evaluation_cost5, evaluation_accuracy5, \
training_cost5, training_accuracy5 = \
    net5.SGD(training_data, epochs=50, mini_batch_size=10, eta=0.01,
             evaluation_data=validation_data, lmbda=1.0,
             monitor_evaluation_accuracy=True)
evaluation_cost6, evaluation_accuracy6, \
training_cost6, training_accuracy6 = \
    net6.SGD(training_data, epochs=50, mini_batch_size=10, eta=0.01,
             evaluation_data=validation_data, lmbda=1.0,
             monitor_evaluation_accuracy=True)
# 开始画图
x = range(1, len(evaluation_accuracy1) + 1)
plt.title('Result Analysis')
plt.plot(x, evaluation_accuracy1, color='green', label='evaluation_accuracy1')
plt.plot(x, evaluation_accuracy2, color='red', label='evaluation_accuracy2')
plt.plot(x, evaluation_accuracy3, color='skyblue', label='evaluation_accuracy3')
plt.plot(x, evaluation_accuracy4, color='blue', label='evaluation_accuracy4')
plt.plot(x, evaluation_accuracy5, color='dimgray', label='evaluation_accuracy5')
plt.plot(x, evaluation_accuracy6, color='magenta', label='evaluation_accuracy6')
plt.legend()  # 显示图例

plt.xlabel('Epoch')
plt.ylabel('rate')
plt.show()
