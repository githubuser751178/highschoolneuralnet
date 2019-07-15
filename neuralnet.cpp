#include "neuralnet.hpp"

void print_v(vector<nntype> v){
	for(nntype x : v){
		cout << x << " ";
	} cout << endl;
}

neural_net::neural_net(int s, nntype ss, nntype diff){
	step_size = ss;
	matrix placeholder1(s, 784);
	matrix placeholder2(10, s);
	placeholder1.randomize();
	placeholder2.randomize();
	weights.push_back(placeholder1);
	weights.push_back(placeholder2);
	corrections.push_back(placeholder1);
	corrections.push_back(placeholder2);
	corrections[0].zero();
	corrections[1].zero();
	differential = diff;
	weights_changed = false;
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
	input = weights[0].timesV(input);
	layer_1_output = input;
	input = weights[1].timesV(input);
	layer_2_output = input;
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
(const vector<nntype>& input, const vector<nntype>& output, const vector<nntype>& target, int n, int r, int c){
	/*
	   cout << "output" << endl;
	   print_v(output); 
	   cout << "target" << endl;
	   print_v(target);
	   p(dlogistic(full_connect_output[r]));
	   */
	//return (output[r] - target[r]) * dlogistic(full_connect_output[r]) * input[c];
	if(n == 1){
		return (output[r] - target[r]) * dlogistic(layer_2_output[r]) * layer_1_output[c];
	} else {
		nntype sum = 0;
		for(int i = 0; i < 10; i++){
			sum += (output[i] - target[i]) * dlogistic(layer_2_output[i]) * weights[1].m[i][r];
		}
		return input[c] * sum;
	}
}

void neural_net::learn(const vector<nntype>& input, const vector<nntype>& output, const vector<nntype>& target){
	//updates corrections for one training image
	int counter = 0, total = 0;
	for(int n = 0; n < weights.size(); n++){
		for(int i = 0; i < weights[n].rows; i++){
			for(int j = 0; j < weights[n].columns; j++){
				nntype partial = partial_derivative(input, output, target, n, i, j);
				/*
				   cout << "layer: " << i << endl;
				   cout << "partial num: " << partial_num << endl;
				   cout << "partial sym: " << partial << endl;
				   cout << "percent err: " << percent_error(partial, partial_num) << endl;
				   cout << " ----------- " << endl;
				   */
				corrections[n].set_element(i, j, corrections[n].e(i, j) - (step_size * partial));
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
	for(int i = 0; i < corrections.size(); i++){
		corrections[i].zero();
	}

	for (int i = 0; i < batch.size(); i++){
		vector<nntype> output = activation(batch[i].pixels);
		if (get_digit(output) == batch[i].label)
			correct += 1;
		learn(batch[i].pixels, output, get_vector(batch[i].label));
	}
		
	for(int i = 0; i < corrections.size(); i++){
		weights[i] = weights[i].plus(corrections[i]);
	}	

	cout << "training percent correct: " << (100 * (correct / batch.size())) << endl;
	cout << "sum of corrections[0]: " << corrections[0].sum_elements() << endl;
	cout << "sum of corrections[1]: " << corrections[1].sum_elements() << endl;
	/*
	cout << "sum of weights: "
		<< weights[0].sum_elements() + weights[1].sum_elements()
		<< endl;
	cout << "total num weights: "
		<< weights[0].num_elements() + weights[1].num_elements()
		<< endl;
	*/
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

nntype neural_net::avg_error(vector<train_img> batch){
	nntype total_error = 0;
	for(int i = 0; i < batch.size(); i++){
		total_error += (error_datum(batch[i].pixels, get_vector(batch[i].label)));
	}
	return total_error / batch.size();
}	
