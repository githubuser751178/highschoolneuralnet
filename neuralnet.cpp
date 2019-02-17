#include <iostream>
#include <vector>
#include <random>
#include "neuralnet.hpp"
using namespace std;

typedef double nntype;

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
	if(x >= 0) return x; else return 0;
}
vector<nntype> neural_net::activation(vector<nntype> input){
	vector<nntype> current = input;
	for(int i = 0; i < weights.size(); i++){
		current = weights[i].timesV(current);
		if (i != weights.size() - 1){
			for(int j = 0; j < current.size(); j++){
				current[j] = ReLU(current[j]);
			}
		}
		if (i == 0) between_layers = current;
	}
	return current;
}
nntype neural_net::error_datum(vector<nntype> input, vector<nntype> target){
	nntype error = 0;
	vector<nntype> output = activation(input);
	for(int i = 0; i < target.size(); i++){
		error += pow(target[i] - output[i], 2);
	}
	return error;
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
	//only works with 1 hidden layer
	vector<nntype> output = activation(x);
	nntype to_return = 0;
	if (weight_m == 1){
		//cout << "reached second weight matrix" << endl;
		to_return = between_layers[c];
		to_return = 2 * (output[r] - target[r]) * to_return;
	}
	else {
		for(int i = 0; i < 10; i++){
			to_return += weights[0].m[i][r] * (output[i] - target[i]);
		}
		nntype ReLU_deriv = 0;
		if(between_layers[r] > 0) ReLU_deriv = 1;
		to_return = ReLU_deriv * x[c] * to_return;
	}
	return to_return;
}

void neural_net::learn(vector<nntype> x, vector<nntype> target){
	//updates corrections for one training image
	int counter = 0, total = 0;
	for(int i = 0; i < weights.size(); i++){
		for(int j = 0; j < weights[i].rows; j++){
			for(int k = 0; k < weights[i].columns; k++){
				nntype partial_num = partial_derivative_num(x, target, weights[i].m[j][k]);
				nntype partial = partial_derivative(x, target, i, j, k);
				if (abs(partial - partial_num) < 1) counter += 1;
				corrections[i].set_element(j, k, corrections[i].e(j, k) - (step_size * partial));
				total += 1;
			}
		}
	}
	cout << "counter " << counter << endl;
	cout << "total " << total << endl; 
}
int neural_net::vectordigit (vector<nntype> output){
	nntype max = *max_element(output.begin(), output.end());
	for (int i = 0; i < output.size(); i++){
		if (output[i] == max)
			return i;
	}
	return -1;
}

vector<nntype> neural_net::digitvector (nntype digit){
	vector<nntype> v;
	for (int i = 0; i < 10; i++) {
		v.push_back(i == digit);
	}
	return v;
}
int neural_net::identify (vector<nntype> pixels){
	return vectordigit(activation(pixels));
}
void neural_net::train (vector<train_img> batch){
	nntype correct = 0;
	for (int i = 0; i < corrections.size(); i++)
		corrections[i].zero();
	for (int i = 0; i < batch.size(); i++){
		vector<nntype> target;
		target = digitvector (batch[i].label);
		if (identify(batch[i].pixels) == batch[i].label)
			correct += 1;
		learn(batch[i].pixels, target);
		cout << "one img processsed" << endl;
	}
	cout << "Percent Correct: " << (100 * (correct / batch.size())) << endl;
	for (int i = 0; i < corrections.size(); i++)
		weights[i] = weights[i].plus(corrections[i]);
}
nntype neural_net::test (vector<train_img> batch){
	nntype correct = 0;
	for (int i = 0; i < batch.size(); i++){
		vector<nntype> target;
		target = digitvector (batch[i].label);
		if (identify(batch[i].pixels) == batch[i].label)
			correct += 1;
	}
	return (100 * (correct / batch.size()));
}
