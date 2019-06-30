#include <iostream>
#include <vector>
#include <random>
#include "neuralnet.hpp"
#include <assert.h>
#include <math.h>

using namespace std;
// foo
typedef double nntype;
typedef vector<nntype> nntype_vector;

#define p(x) cout << #x << ": " << x << endl

/*
//nn_map is a memoization table
void nn_map::insert(vector<nntype> key, vector<nntype> value){
	map.push_back(make_pair(key, value));
}
vector<nntype> nn_map::at(vector<nntype> key){
	for(int i = 0; i < map.size(); i++){
		if(map[i].first == key)
			return map[i].second;
	}
	return {0}; //just here to get rid of compiler warning
}
bool nn_map::contains(vector<nntype> key){
	for(int i = 0; i < map.size(); i++){
		if(map[i].first == key)
			return true;
	}
	return false;
}
*/

neural_net::neural_net(int i, nntype ss, nntype diff){
	inputs = i;
	step_size = ss;
	differential = diff;
	weights_changed = false;
	weight_cap = 5;
	matrix blankweight(10, 784);
	blankweight.randomize();
	weights.push_back(blankweight);
	corrections.push_back(blankweight);
}

nntype neural_net::logistic(nntype x){
	return 1.0 / (1.0 + exp( -x));
}

nntype neural_net::dlogistic(nntype x){
	return exp(-x) / pow(1 + exp(-x), 2);
}

vector<nntype> neural_net::activation(vector<nntype> input){
	input = weights[0].timesV(input);
	full_connect_output = input;
	for(int i = 0; i < input.size(); i++){
		input[i] = logistic(input[i]);
	}
	return input;
}

nntype neural_net::error_datum(vector<nntype> input, vector<nntype> target){
	nntype error = 0;
	vector<nntype> output = activation(input);
	for(int i = 0; i < target.size(); i++){
		error += pow(target[i] - output[i], 2);
	}
	return 0.5 * error;
}
nntype neural_net::error_data
(vector< vector<nntype> > inputs, vector< vector<nntype> > targets){
	nntype total_e = 0;
	for(int i = 0; i < inputs.size(); i++)
		total_e += error_datum(inputs[i], targets[i]);
	return total_e;
}

nntype neural_net::partial_derivative_num (vector<nntype> input, vector<nntype> target, nntype &weight){
	nntype error = error_datum(input, target);
	weight += differential;
	weights_changed = true;
	nntype error_plus_h = error_datum(input, target);
	weight -= differential;
	weights_changed = false;
	return (error_plus_h - error) / differential;
}

void print_v(vector<nntype> v){
	for(nntype x : v){
		cout << x << " ";
	} cout << endl;
}


nntype neural_net::partial_derivative
(const vector<nntype>& input, const vector<nntype>& output, const vector<nntype>& target, int weight_m, int r, int c){
	cout << "output" << endl;
	print_v(output); 
	cout << "target" << endl;
	print_v(target);
	p(dlogistic(full_connect_output[r]));
	return (output[r] - target[r]) * dlogistic(full_connect_output[r]) * input[c];
}

void neural_net::learn(const vector<nntype>& input, const vector<nntype>& output, const vector<nntype>& target){
	//updates corrections for one training image
	int counter = 0, total = 0;
	for(int i = 0; i < weights.size(); i++){
		for(int j = 0; j < weights[i].rows; j++){
			for(int k = 0; k < weights[i].columns; k++){
				//nntype partial_num = partial_derivative_num(x, target, weights[i].m[j][k]);
				nntype partial = partial_derivative(input, output, target, i, j, k);
				cout << "partial: " << partial << endl;
				/*
				   cout << "layer: " << i << endl;
				   cout << "partial num: " << partial_num << endl;
				   cout << "partial sym: " << partial << endl;
				   cout << "percent err: " << percent_error(partial, partial_num) << endl;
				   cout << " ----------- " << endl;
				*/
				corrections[i].set_element(j, k, corrections[i].e(j, k) - (step_size * partial));
			}
		}
	}
}
//if index 1 is bigger, return true
int neural_net::get_digit(vector<nntype> output){
	int counter = 0;
	for(int i = 1; i < 10; i++){
		if(output[i] > output[counter]){
			counter = i;
		}
	}
	return counter;
}

vector<nntype> neural_net::get_vector(int digit){
	vector<nntype> v;
	for(int i = 0; i < 10; i++)
		v.push_back(0);
	v[digit] = 1;
	return v;
}

int neural_net::identify (vector<nntype> pixels){
	return get_digit(activation(pixels));
}
void neural_net::train (vector<train_img> batch){
	nntype correct = 0;
	for (int i = 0; i < corrections.size(); i++)
		corrections[i].zero();
	for (int i = 0; i < batch.size(); i++){
		//cout << "batch i " << i << endl;
		vector<nntype> output = activation(batch[i].pixels);
		if (get_digit(output) == batch[i].label)
			correct += 1;
		cout << "error: " << error_datum(batch[i].pixels, get_vector(batch[i].label)) << endl;
		learn(batch[i].pixels, output, get_vector(batch[i].label));
		cout << "sum elements of corrections: " << corrections[0].sum_elements() << endl;
		cout << "full connect output: ";
		for(float j : full_connect_output){
			cout << j << " ";
		} cout << endl;
		cout << "output: ";
		for(float j : output){
			cout << j << " ";
		} cout << endl;
		p(batch[i].label);
		cout << "learned on image #" << i << endl;
	}
	cout << "training percent correct: " << (100 * (correct / batch.size())) << endl;
	for (int i = 0; i < corrections.size(); i++) {
		//assert (corrections[i].sum_elements() != 0);
		weights[i] = weights[i].plus(corrections[i]);
	//memo.clear();
	}
	cout << "corrections[1]: " << endl;
	corrections[0].print(1,10);
	cout << "weights[1]: " << endl;
	weights[0].print(1,10);
}

nntype neural_net::test (vector<train_img> batch){
	nntype correct = 0;
	for (int i = 0; i < batch.size(); i++){
		if (identify(batch[i].pixels) == batch[i].label)
			correct += 1;
	}
	return (100 * (correct / batch.size()));
}

