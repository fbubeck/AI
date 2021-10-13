from algorithms import TensorFlow
from algorithms import LinearRegression
from algorithms import DecisionTree
from algorithms import RandomForestRegressor
from data import SampleData
from data import Exploration
import json


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
    tensorFlow.train()
    tensorFlow.test()

    # Start Linear Regression
    linearRegression.train()
    linearRegression.test()

    # Start Decision Tree
    decisionTree.train()
    decisionTree.test()

    # Start Random Forest
    randomForest.train()
    randomForest.test()


if __name__ == "__main__":
    main()
