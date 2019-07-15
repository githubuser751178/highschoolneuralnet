#include "neuralnet.hpp"
#include <cmath>
#include <chrono>

//NEURAL NET CONFIG---------------------------------
bool CHECK_PARTIALS = true;
int NUM_CHECKS = 10;
int NN_SHAPE = 10;
int TRAIN_FILE_SIZE = 50000, TEST_FILE_SIZE = 9000;
int TRAIN_SIZE = 1000, TEST_SIZE = 100, EPOCH = 15;
nntype LEARNING_RATE = .010, APPROX_H = .001;
//--------------------------------------------------

randp::randp(int n){
	for(int i = 0; i < n; i++){
		nums.push_back(i);
	}
}

int randp::next_int(){
	if(nums.size() == 0)
		return -1;
	srand(time(NULL));
	int index = (rand() % nums.size());
	int num = nums[index];
	nums.erase(nums.begin() + index);
	return num;
}

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

bool partials_match(int NUM_CHECKS){
	//return true;
	
	//randomly generates data set to check symbolic partial
	srand(time(NULL));
	neural_net partial_check(5, .1, .01);
	for(int i = 0; i < NUM_CHECKS; i++){
		//input output
		vector<nntype> dummy_input, dummy_target;
		for(int j = 0; j < 784; j++)
			dummy_input.push_back((rand() % 2) / 255.0);
		for(int j = 0; j < 10; j++)
			dummy_target.push_back(0);
		dummy_target[rand() % 10] = 1;

		//determine whether to check partial on a inner or outer weight
		int n = rand() % 2;
		nntype partial_num = partial_check.partial_derivative_num
			(dummy_input, dummy_target, partial_check.weights[n].m[0][0]);
		nntype partial_sym = partial_check.partial_derivative
			(dummy_input, partial_check.activation(dummy_input), dummy_target, n, 0, 0);
		if(percent_error(partial_sym, partial_num) > .01){
			cout << "partials do not match" << endl;
			cout << "partial_sym: " << partial_sym << endl;
			cout << "partial_num: " << partial_num << endl;
			cout << "% error: " << percent_error(partial_sym, partial_num) << endl; 
			return false;
		}
	}
	cout << "partials match" << endl;
	return true;
	
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
	neural_net solve_mnist(NN_SHAPE, LEARNING_RATE, APPROX_H);
	if(CHECK_PARTIALS){ 
		partials_match(NUM_CHECKS); 
	}

	vector<train_img> training_set, test_set, entire_train, entire_test;

	entire_train = read_mnist("mnist_train.csv", TRAIN_FILE_SIZE);
	entire_test = read_mnist("mnist_test.csv", TEST_FILE_SIZE);
	cout << "file read complete" << endl;

	test_set = get_batch(TEST_SIZE, entire_test);
	cout << "test percent correct before training: " << solve_mnist.test(test_set) << endl;

	chrono::steady_clock::time_point begin = chrono::steady_clock::now();

	for(int i = 0; i < EPOCH; i++){
		training_set = get_batch(TRAIN_SIZE, entire_train);
		solve_mnist.train(training_set);
		cout << "average error: " << solve_mnist.avg_error(training_set) << endl;
		//cout << "test percent correct after training: " << solve_mnist.test(test_set) << endl;
	}
	chrono::steady_clock::time_point end = chrono::steady_clock::now();

	cout << "Time difference (seconds): " 
		<< chrono::duration_cast<chrono::microseconds>
		(end - begin).count() / 1000000.0 << endl;
	cout << "test percent correct after training: " << solve_mnist.test(test_set) << endl;

	return 0;
}
