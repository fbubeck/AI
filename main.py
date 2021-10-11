import TensorFlow
import LinearRegression


def main():
    tensorFlow = TensorFlow.TensorFlow(50000)
    linearRegression = LinearRegression.LinearRegression(50000)

    tensorFlow.train()
    tensorFlow.test()

    linearRegression.train()
    linearRegression.test()


if __name__ == "__main__":
    main()
