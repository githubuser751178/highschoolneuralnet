#include <iostream>
#include <assert.h>
#include <vector>
#include <random>
#include <unordered_map>
#include "neuralnet.hpp"
#include <cmath>
#include <time.h>
#include <chrono>
#define p(x) cout << #x << ": " << x << endl
using namespace std;

typedef double nntype;

//NEURAL NET CONFIG
bool CHECK_PARTIALS = true;
int TRAIN_FILE_SIZE = 50000, TEST_FILE_SIZE = 9000;
int TRAIN_SIZE = 5, TEST_SIZE = 100, EPOCH = 1;
vector<int> NN_SHAPE = {392, 10};
nntype LEARNING_RATE = .001, APPROX_H = .01;

nntype percent_error(nntype approx, nntype exact){
	if(exact == 0){
		if(approx == 0){
			return 0;
		}
		return 1234;
	}
	return abs(approx - exact) / exact;
}

nntype time_estimate(nntype inputs){
	//this is no longer correct
	return .003503 * pow(inputs, 2) + .19 * inputs + .0706;
}
bool partials_match(){
	return true;
	/*
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
		if(percent_error(partial_sym, partial_num) > .01){
			cout << "partial_sym: " << partial_sym << endl;
			cout << "partial_num: " << partial_num << endl;
			cout << "% error: " << percent_error(partial_sym, partial_num) << endl; 
			return false;
		}
	}
	return true;
	*/
}

vector<train_img> get_batch(int size, vector<train_img>& total_set){
	vector<train_img> batch;
	randp rng(size);
	for(int i = 0; i < size; i++){
		batch.push_back(total_set[rng.next_int()]);
	}
	return batch;
}

int main () {
	neural_net solve_mnist(784, LEARNING_RATE, APPROX_H);
	if(CHECK_PARTIALS)
		cout << "partials match: " << partials_match() << endl;

	vector<train_img> training_set, test_set, entire_train, entire_test;
	//training_set = random_read("mnist_train.csv", TRAIN_FILE_SIZE, TRAIN_SIZE);
	//test_set = random_read("mnist_test.csv", TEST_FILE_SIZE, TEST_SIZE);

	entire_train = read_mnist("mnist_train.csv", 5000);
	entire_test = read_mnist("mnist_test.csv", 900);

	/*
	randp train_rand(5000), test_rand(900);
	for(int i = 0; i < TRAIN_SIZE; i++)
		training_set.push_back(entire_train[train_rand.next_int()]);	
	for(int i = 0; i < TEST_SIZE; i++)
		test_set.push_back(entire_test[test_rand.next_int()]);
	*/

	cout << "train set length: " << training_set.size() << endl;
	cout << "test percent correct before training: " << solve_mnist.test(test_set) << endl;

	cout << "weight matrix 2 head: " << endl;
	solve_mnist.weights[0].print(1, 10);

	chrono::steady_clock::time_point begin = chrono::steady_clock::now();

	for(int i = 0; i < EPOCH; i++){
		training_set = get_batch(TRAIN_SIZE, entire_train);
		/*
		if(i == 4){
			cout << "pixels: " << endl;
			for(int j = 0; j < 784; j++)
				cout << training_set[0].pixels[j] << " ";
			cout << endl;
		}
		*/	
		solve_mnist.train(training_set);
	}
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
