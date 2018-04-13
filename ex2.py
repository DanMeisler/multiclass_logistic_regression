import matplotlib.pyplot as plt
import numpy as np


class MulticlassLogisticRegression(object):

    NUM_OF_FEATURES = 1
    NUM_OF_LEARNING_EPOCHS = 300
    LEARNING_RATE = 0.01

    def __init__(self, classes):
        self.classes = classes
        self.b = np.random.rand(len(self.classes), 1)
        self.w = np.random.rand(len(self.classes), self.NUM_OF_FEATURES)
        self.num_of_mistakes = 0
        self.total_iterations = 0

    @staticmethod
    def softmax(x):
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)

    def calc_loss(self, yt, y_hat):
        self.total_iterations += 1
        if yt != y_hat:
            self.num_of_mistakes += 1

        return float(self.num_of_mistakes) / float(self.total_iterations)

    def w_gradient(self, xt, yt):
        scores = self.w.dot(xt) + self.b
        normalized_scores = self.softmax(scores)
        normalized_scores[self.classes.index(yt)] -= 1
        return normalized_scores.dot(xt)

    def b_gradient(self, xt, yt):
        scores = self.w.dot(xt) + self.b
        normalized_scores = self.softmax(scores)
        normalized_scores[self.classes.index(yt)] -= 1
        return normalized_scores

    def update_parameters(self, xt, yt):
        self.w -= self.LEARNING_RATE * self.w_gradient(xt, yt)
        self.b -= self.LEARNING_RATE * self.b_gradient(xt, yt)

    def learn(self, training_set):
        for epoch_number in xrange(self.NUM_OF_LEARNING_EPOCHS):
            np.random.shuffle(training_set)
            for xt, yt in training_set:
                y_hat = self.predict(xt)
                print self.calc_loss(yt, y_hat)
                self.update_parameters(xt, yt)

    def probability_vector(self, xt):
        return self.softmax(np.dot(self.w, xt) + self.b)

    def predict(self, xt):
        return self.classes[np.argmax(self.probability_vector(xt))]


def create_normal_pdf(mean, variance):
    def normal_pdf(x):
        return np.exp(-np.square(x - mean) / 2 * variance) / (np.sqrt(2 * np.pi * variance))

    return normal_pdf


def generate_training_set():
    data = []
    for a in [1, 2, 3]:
        x = np.random.normal(2 * a, 1, 100)
        data += zip(x, [a] * len(x))

    return data


def generate_test_set():
    test_set = np.array([])
    for a in [1, 2, 3]:
        test_set = np.append(test_set, np.random.normal(2 * a, 1, 300))

    return test_set


if __name__ == "__main__":

    mlr = MulticlassLogisticRegression([1, 2, 3])
    mlr.learn(generate_training_set())

    test_set = generate_test_set()

    f_y1 = create_normal_pdf(2, 1)
    f_y2 = create_normal_pdf(4, 1)
    f_y3 = create_normal_pdf(6, 1)

    plt.axis((0, 10, 0, 1))
    plt.plot(test_set, f_y1(test_set) / (f_y1(test_set) + f_y2(test_set) + f_y3(test_set)),
             "ro", label="real distribution")
    plt.plot(test_set, [mlr.probability_vector(test_set[i])[0] for i in xrange(len(test_set))],
             "bo", label="prediction")

    plt.legend()
    plt.show()
