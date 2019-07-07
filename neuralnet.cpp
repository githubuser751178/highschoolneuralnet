#include "neuralnet.hpp"

void print_v(vector<nntype> v){
	for(nntype x : v){
		cout << x << " ";
	} cout << endl;
}

neural_net::neural_net(nntype ss, nntype diff) : weights(10, 784), corrections(10, 784){
	step_size = ss;
	differential = diff;
	weights_changed = false;
	weights.randomize();
	corrections.zero();
}

//sigmoidal non linear operator
nntype neural_net::logistic(nntype x){
	return 1.0 / (1.0 + exp( -x));
}

//derivative of logistic function
nntype neural_net::dlogistic(nntype x){
	return exp(-x) / pow(1 + exp(-x), 2);
}

vector<nntype> neural_net::activation(vector<nntype> input){
	input = weights.timesV(input);
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

nntype neural_net::partial_derivative
(const vector<nntype>& input, const vector<nntype>& output, const vector<nntype>& target, int r, int c){
	/*
	   cout << "output" << endl;
	   print_v(output); 
	   cout << "target" << endl;
	   print_v(target);
	   p(dlogistic(full_connect_output[r]));
	   */
	return (output[r] - target[r]) * dlogistic(full_connect_output[r]) * input[c];
}

void neural_net::learn(const vector<nntype>& input, const vector<nntype>& output, const vector<nntype>& target){
	//updates corrections for one training image
	int counter = 0, total = 0;
	for(int i = 0; i < weights.rows; i++){
		for(int j = 0; j < weights.columns; j++){
			nntype partial = partial_derivative(input, output, target, i, j);
			/*
			   cout << "layer: " << i << endl;
			   cout << "partial num: " << partial_num << endl;
			   cout << "partial sym: " << partial << endl;
			   cout << "percent err: " << percent_error(partial, partial_num) << endl;
			   cout << " ----------- " << endl;
			   */
			corrections.set_element(i, j, corrections.e(i, j) - (step_size * partial));
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
	corrections.zero();
	for (int i = 0; i < batch.size(); i++){
		vector<nntype> output = activation(batch[i].pixels);
		if (get_digit(output) == batch[i].label)
			correct += 1;
		learn(batch[i].pixels, output, get_vector(batch[i].label));
	}
	weights = weights.plus(corrections);
	/*
	cout << "training percent correct: " << (100 * (correct / batch.size())) << endl;
	cout << "sum of corrections: " << corrections.sum_elements() << endl;
	*/
	//cout << "sum of weights: " << weights.sum_elements() << endl;
	//cout << "weights updated \n";
	
}

nntype neural_net::test (vector<train_img> batch){
	nntype correct = 0;
	for (int i = 0; i < batch.size(); i++){
		if (identify(batch[i].pixels) == batch[i].label)
			correct += 1;
	}
	return (100 * (correct / batch.size()));
}

