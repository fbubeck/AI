from algorithms import TensorFlow
from algorithms import LinearRegression
from algorithms import DecisionTree
from algorithms import RandomForestRegressor
from data import SampleData
from data import Exploration
import json
from matplotlib import pyplot as plt


def main():
    # read config.json
    with open('config/config.json') as file:
        config = json.load(file)

    # Get Global Parameters from config file
    n_numbers = config["GlobalParameters"]["n_numbers"]
    min_bias = config["GlobalParameters"]["min_bias"]
    max_bias = config["GlobalParameters"]["max_bias"]

    # Get Sample Data
    sampleData = SampleData.SampleData(n_numbers, min_bias, max_bias)
    train_data = sampleData.get_Data()
    test_data = sampleData.get_Data()

    # Data Exploration
    exploration = Exploration.Exploration(train_data, test_data)
    exploration.analyseData()

    # Creating Objects
    tensorFlow = TensorFlow.TensorFlow(train_data, test_data)
    linearRegression = LinearRegression.LinearRegression(train_data, test_data)
    decisionTree = DecisionTree.DecisionTree(train_data, test_data)
    randomForest = RandomForestRegressor.RandomForest(
        train_data, test_data)

    # Start Tensorflow
    tensorFlow_training = tensorFlow.train()
    tensorFlow_test = tensorFlow.test()

    # Start Linear Regression
    linearRegression_training = linearRegression.train()
    linearRegression_test = linearRegression.test()

    # Start Decision Tree
    decisionTree_training = decisionTree.train()
    decisionTree_test = decisionTree.test()

    # Start Random Forest
    randomForest_training = randomForest.train()
    randomForest_test = randomForest.test()

    # Plot Summary
    fig = plt.figure()
    plt.ylabel('Training Duration in seconds (log scale)')
    plt.title('Training Comparison')
    plt.yscale('log')
    x = ['TensorFlow', 'linearRegression', 'decisionTree', 'randomForest']
    y = [tensorFlow_training, linearRegression_training,
         decisionTree_training, randomForest_training, ]
    plt.bar(x, y)
    plt.show()


if __name__ == "__main__":
    main()
