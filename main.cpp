#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>
#include "neuralnet.hpp"
#include <chrono>

using namespace std;

typedef double nntype;

nntype percent_error(nntype approx, nntype exact){
	return abs(approx - exact) / exact;
}
nntype time_estimate(nntype inputs){
	return .003503 * pow(inputs, 2) + .19 * inputs + .0706;
}
int main () {
	vector<train_img> training_set, test_set;
	training_set = read_mnist("mnist_train.csv", 200);
	cout << "n^2 time estimate: " << time_estimate(60000) << endl;
	cout << "train set len: " << training_set.size() << endl;
	test_set = read_mnist("mnist_test.csv", 100);
	vector<int> shapes = {392, 10};
	neural_net myXOR(shapes, 2, .1, .01);
	//cout << "weight matrices" << endl;
	//myXOR.weights[0].print();
	//myXOR.weights[1].print();
	cout << "test percent correct before training: " << myXOR.test(test_set) << endl;
	chrono::steady_clock::time_point begin = chrono::steady_clock::now();
	myXOR.train(training_set);
	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	cout << "Time difference (seconds): " 
		<< chrono::duration_cast<chrono::microseconds>
		(end - begin).count() / 1000000.0 << endl;
	cout << "test percent correct after training: " << myXOR.test(test_set) << endl;
	//cout << "weight matrices" << endl;
	//myXOR.weights[0].print();
	//myXOR.weights[1].print();
	return 0;
}
