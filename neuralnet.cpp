#include <iostream>
#include <vector>
#include <random>
#include "neuralnet.hpp"
using namespace std;

typedef double nntype;
typedef vector<nntype> nntype_vector;

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

neural_net::neural_net(vector<int> s, int i, nntype ss, nntype diff){
	shape = s;
	inputs = i;
	step_size = ss;
	differential = diff;
	weights_changed = false;
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
	/*
	if(!weights_changed && memo.find(input) != memo.end())
		return memo.at(input);
	*/
	for(int i = 0; i < weights.size(); i++){
		current = weights[i].timesV(current);
		if (i != weights.size() - 1){
			for(int j = 0; j < current.size(); j++){
				current[j] = ReLU(current[j]);
			}
		}
		if (i == 0) between_layers = current;
	}
	//activationMemo.insert(input, current);
	//memo.emplace(input, current);
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

nntype neural_net::partial_derivative_num (vector<nntype> input, vector<nntype> target, nntype &weight){
	nntype error = error_datum(input, target);
	weight += differential;
	weights_changed = true;
	nntype error_plus_h = error_datum(input, target);
	weight -= differential;
	weights_changed = false;
	return (error_plus_h - error) / differential;
}
nntype neural_net::partial_derivative
(vector<nntype> input, vector<nntype> output, vector<nntype> target, int weight_m, int r, int c){
	if (weight_m == 1){
		return (output[r] - target[r]) * between_layers[c];
	}
	else {
		if (dot(weights[0].m[r], input) <= 0)
			return 0;
		nntype summation = 0;
		for(int n = 0; n < target.size(); n++)
			summation += (output[n] - target[n]) * weights[1].m[n][r];
		return summation * input[c];
	}
}

void neural_net::learn(vector<nntype> input, vector<nntype> output, vector<nntype> target){
	//updates corrections for one training image
	int counter = 0, total = 0;
	for(int i = 0; i < weights.size(); i++){
		for(int j = 0; j < weights[i].rows; j++){
			for(int k = 0; k < weights[i].columns; k++){
				//nntype partial_num = partial_derivative_num(x, target, weights[i].m[j][k]);
				nntype partial = partial_derivative(input, output, target, i, j, k);
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
		vector<nntype> output = activation(batch[i].pixels);
		if (get_digit(output) == batch[i].label)
			correct += 1;
		learn(batch[i].pixels, output, get_vector(batch[i].label));
	}
	cout << "training percent correct: " << (100 * (correct / batch.size())) << endl;
	for (int i = 0; i < corrections.size(); i++)
		weights[i] = weights[i].plus(corrections[i]);
	//memo.clear();
}
nntype neural_net::test (vector<train_img> batch){
	nntype correct = 0;
	for (int i = 0; i < batch.size(); i++){
		if (identify(batch[i].pixels) == batch[i].label)
			correct += 1;
	}
	return (100 * (correct / batch.size()));
}

