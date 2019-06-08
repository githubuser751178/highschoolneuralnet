#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>
#include "neuralnet.hpp"
#include <cmath>
#include <time.h>
#include <chrono>

using namespace std;

typedef double nntype;

nntype percent_error(nntype approx, nntype exact){
	return abs(approx - exact) / exact;
}
nntype time_estimate(nntype inputs){
	return .003503 * pow(inputs, 2) + .19 * inputs + .0706;
}
bool partials_match(){
	//randomly generates data set for 
	int DIM1 = 3, DIM2 = 2, NUM_CHECKS = 10;
	srand(time(NULL));
	vector<int> shapes = {DIM1, DIM2};
	neural_net partial_check(shapes, 3, .1, .01);
	for(int i = 0; i < NUM_CHECKS; i++){
		vector<nntype> dummy_input, dummy_target;
		for(int j = 0; j < DIM1; j++)
			dummy_input.push_back(rand() % 2);	
		for(int j = 0; j < DIM2; j++)
			dummy_target.push_back(rand() % 2);
		nntype partial_num = partial_check.partial_derivative_num
			(dummy_input, dummy_target, partial_check.weights[0].m[0][0]);
		nntype partial_sym = partial_check.partial_derivative
			(dummy_input, partial_check.activation(dummy_input), dummy_target, 0, 0, 0);
		if(percent_error(partial_sym, partial_num) > .01)
			return false;
	}
	return true;
}

int main () {
	//change test set to random
	vector<train_img> training_set, test_set;
	cout << "partials match? " << partials_match() << endl;
	training_set = read_mnist("mnist_train.csv", 10);
	cout << "n^2 time estimate: " << time_estimate(60000) << endl;
	cout << "train set len: " << training_set.size() << endl;
	test_set = read_mnist("mnist_test.csv", 10);
	vector<int> shapes = {392, 10};
	neural_net please_work(shapes, 784, .1, .01);
	//cout << "weight matrices" << endl;
	//myXOR.weights[0].print();
	//myXOR.weights[1].print();
	cout << "test percent correct before training: " << please_work.test(test_set) << endl;
	chrono::steady_clock::time_point begin = chrono::steady_clock::now();
	please_work.train(training_set);
	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	cout << "Time difference (seconds): " 
		<< chrono::duration_cast<chrono::microseconds>
		(end - begin).count() / 1000000.0 << endl;
	cout << "test percent correct after training: " << please_work.test(test_set) << endl;
	//cout << "weight matrices" << endl;
	//myXOR.weights[0].print();
	//myXOR.weights[1].print();
	return 0;
}
