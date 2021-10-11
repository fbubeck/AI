import TensorFlow
import LinearRegression
import SampleData


def main():
    # Get Sample Data
    n_numbers = 50000
    sampleData = SampleData.SampleData(n_numbers)
    train_data = sampleData.get_Data()
    test_data = sampleData.get_Data()

    # Creating Objects
    tensorFlow = TensorFlow.TensorFlow(train_data, test_data)
    linearRegression = LinearRegression.LinearRegression(train_data, test_data)

    # Start Tensorflow
    tensorFlow.train()
    tensorFlow.test()

    # Start Linear Regression
    linearRegression.train()
    linearRegression.test()


if __name__ == "__main__":
    main()
