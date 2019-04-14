#include <iostream>
#include <unordered_map>
#include <vector>
#include <random>
#include "neuralnet.hpp"
using namespace std;

typedef double nntype;

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

neural_net::neural_net(vector<int> s, int i, nntype ss, nntype diff){
	shape = s;
	inputs = i;
	step_size = ss;
	differential = diff;
	matrix blankweight1 (inputs, shape[0]);
	blankweight1.randomize();
	weights.push_back(blankweight1);
	corrections.push_back(blankweight1);
	for(int i = 1; i < shape.size(); i++){
		matrix blankweight (shape[i - 1], shape[i]);
		blankweight.randomize();
		weights.push_back(blankweight);
		corrections.push_back(blankweight);
	}
}
nntype neural_net::ReLU(nntype x){
	if(x >= 0) 
		return x; 
	return 0;
}

vector<nntype> neural_net::activation(vector<nntype> input){
	vector<nntype> current = input;
	if(activationMemo.contains(input))
		return activationMemo.at(input);
	for(int i = 0; i < weights.size(); i++){
		current = weights[i].timesV(current);
		if (i != weights.size() - 1){
			for(int j = 0; j < current.size(); j++){
				current[j] = ReLU(current[j]);
			}
		}
		if (i == 0) between_layers = current;
	}
	activationMemo.insert(input, current);
	return current;
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

nntype neural_net::partial_derivative_num (vector<nntype> x, vector<nntype> target, nntype &weight){
	nntype error = error_datum(x, target);
	weight += differential;
	nntype error_plus_h = error_datum(x, target);
	weight -= differential;
	return (error_plus_h - error) / differential;
}
nntype neural_net::partial_derivative (vector<nntype> x, vector<nntype> target, int weight_m, int r, int c){
	vector<nntype> output = activation(x);
	if (weight_m == 1){
		return (output[r] - target[r]) * between_layers[c];
	}
	else {
		if (dot(weights[0].m[r], x) <= 0)
			return 0;
		nntype summation = 0;
		for(int n = 0; n < target.size(); n++)
			summation += weights[1].m[n][r];
		return summation * x[c];
	}
}

void neural_net::learn(vector<nntype> x, vector<nntype> target){
	//updates corrections for one training image
	int counter = 0, total = 0;
	for(int i = 0; i < weights.size(); i++){
		for(int j = 0; j < weights[i].rows; j++){
			for(int k = 0; k < weights[i].columns; k++){
				nntype partial_num = partial_derivative_num(x, target, weights[i].m[j][k]);
				nntype partial = partial_derivative(x, target, i, j, k);
				//if (abs(partial - partial_num) < 1) 
				//	counter += 1;
//				cout << partial << endl;
				corrections[i].set_element(j, k, corrections[i].e(j, k) - (step_size * partial));
				//total += 1;
			}
		}
	}
}
//if index 1 is bigger, return true
int neural_net::vectordigit (vector<nntype> output){
	if(output[1] > output[0])
		return 1;
	return 0;
}

int neural_net::identify (vector<nntype> pixels){
	return vectordigit(activation(pixels));
}
void neural_net::train (vector<xor_input> batch){
	nntype correct = 0;
	for (int i = 0; i < corrections.size(); i++)
		corrections[i].zero();
	for (int i = 0; i < batch.size(); i++){
		if (identify(batch[i].things) == batch[i].label)
			correct += 1;
		learn(batch[i].things, batch[i].target);
	}
	cout << "training percent correct: " << (100 * (correct / batch.size())) << endl;
	for (int i = 0; i < corrections.size(); i++)
		weights[i] = weights[i].plus(corrections[i]);
	activationMemo.map.clear();
}
nntype neural_net::test (vector<xor_input> batch){
	nntype correct = 0;
	for (int i = 0; i < batch.size(); i++){
		if (identify(batch[i].things) == batch[i].label)
			correct += 1;
	}
	return (100 * (correct / batch.size()));
}
