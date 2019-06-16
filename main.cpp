#include <iostream>
#include <assert.h>
#include <vector>
#include <random>
#include <unordered_map>
#include "neuralnet.hpp"
#include <cmath>
#include <time.h>
#include <chrono>

using namespace std;

typedef double nntype;

bool CHECK_PARTIALS = true;
int TRAIN_SET_LEN = 50000, TEST_SET_LEN = 9000;
int TRAIN_SIZE = 100, TEST_SIZE = 100;
vector<int> NN_SHAPE = {392, 10};
nntype STEP_SIZE = .1, APPROX_H = .01;

nntype percent_error(nntype approx, nntype exact){
	return abs(approx - exact) / exact;
}
nntype time_estimate(nntype inputs){
	//this is no longer correct
	return .003503 * pow(inputs, 2) + .19 * inputs + .0706;
}
bool partials_match(){
	//randomly generates data set to check symbolic partial
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
	//next task: randomize which images to test on
	if(CHECK_PARTIALS)
		assert(partials_match());

	vector<train_img> training_set, test_set, entire_train, entire_test;
	
	entire_train = read_mnist("mnist_train.csv", 50000);
	entire_test = read_mnist("mnist_test.csv", 9000);
	neural_net solve_mnist(NN_SHAPE, 784, STEP_SIZE, APPROX_H);
	
	randp train_rand(50000), test_rand(9000);
	for(int i = 0; i < TRAIN_SIZE; i++)
		training_set.push_back(entire_train[train_rand.next_int()]);	
	for(int i = 0; i < TEST_SIZE; i++)
		test_set.push_back(entire_test[test_rand.next_int()]);

	cout << "train set len: " << training_set.size() << endl;
	cout << "test percent correct before training: " << solve_mnist.test(test_set) << endl;

	chrono::steady_clock::time_point begin = chrono::steady_clock::now();
	solve_mnist.train(training_set);
	chrono::steady_clock::time_point end = chrono::steady_clock::now();

	cout << "Time difference (seconds): " 
		<< chrono::duration_cast<chrono::microseconds>
		(end - begin).count() / 1000000.0 << endl;

	cout << "test percent correct after training: " << solve_mnist.test(test_set) << endl;
	
	/*
	randp tester(5);
	for(int i = 0; i < 7; i++){
		cout << tester.next_int() << endl;
	}
	*/
	return 0;
}
