from algorithms import TensorFlow
from algorithms import LinearRegression
from algorithms import DecisionTree
from algorithms import RandomForestRegressor
from data import SampleData
from data import Exploration
import json
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


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
    train_data = sampleData.getData()
    test_data = sampleData.getData()

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
    tensorFlow.plot()

    # Start Linear Regression
    linearRegression_training = linearRegression.train()
    linearRegression_test = linearRegression.test()
    linearRegression.plot()

    # Start Decision Tree
    decisionTree_training = decisionTree.train()
    decisionTree_test = decisionTree.test()

    # Start Random Forest
    randomForest_training = randomForest.train()
    randomForest_test = randomForest.test()

    # Plot Summary
    fig = plt.figure()
    plt.ylabel('Training Duration [in seconds]')
    plt.xlabel('Mean Squarred Error')
    plt.title('Training duration and mse of different algorithms')
    plt.scatter(tensorFlow_test[1], tensorFlow_training, s=100, c='blue', alpha=0.4)
    plt.scatter(linearRegression_test[1], linearRegression_training, s=100, c='red', alpha=0.4)
    plt.scatter(decisionTree_test[1], decisionTree_training, s=100, c='green', alpha=0.4)
    plt.scatter(randomForest_test[1], randomForest_training, s=100, c='orange', alpha=0.4)
    plt.legend(["TensorFlow Neural Network", "Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"], loc ="upper left")
    plt.show()

    fig = plt.figure()
    plt.ylabel('Inference Duration [in seconds]')
    plt.xlabel('Mean Squarred Error')
    plt.title('Inference duration and mse of different algorithms')
    plt.scatter(tensorFlow_test[1], tensorFlow_test[0], s=100, c='blue', alpha=0.4)
    plt.scatter(linearRegression_test[1], linearRegression_test[0], s=100, c='red', alpha=0.4)
    plt.scatter(decisionTree_test[1], decisionTree_test[0], s=100, c='green', alpha=0.4)
    plt.scatter(randomForest_test[1], randomForest_test[0], s=100, c='orange', alpha=0.4)
    plt.legend(["TensorFlow Neural Network", "Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"], loc ="upper left")
    plt.show()


if __name__ == "__main__":
    main()
