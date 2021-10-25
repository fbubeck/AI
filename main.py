from algorithms import TensorFlow
from algorithms import LinearRegression
from algorithms import DecisionTree
from algorithms import RandomForestRegressor
from data import SampleData
from data import Exploration
import Evaluation
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


if __name__ == "__main__":
    main()
