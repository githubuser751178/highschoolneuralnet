#include <iostream>
#include <vector>
#include <random>
#include "neuralnet.hpp"
using namespace std;

typedef double nntype;

int main () {
	vector<train_img> training_set;
	vector<train_img> test_set;
	training_set = read_mnist("mnist_train.csv", 1000);
	//test_set = read_mnist("mnist_test.csv");
	vector<train_img> batch1;
	cout << training_set.size() << endl;
	for(int i = 0; i < 2; i++)
		batch1.push_back(training_set[i]);
	vector<int> shapes = {392, 10};
	neural_net beat_mnist(shapes, 784, .1, .01);
	beat_mnist.train(batch1);
	//cout << beat_mnist.weights[0].e(0,0) << endl;
	//cout << beat_mnist.identify(batch1[0].pixels) << endl;
	return 0;
}
