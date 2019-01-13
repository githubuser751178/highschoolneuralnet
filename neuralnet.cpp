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
  for(int i = 1; i < shape.size(); i++){
    matrix blankweight (shape[i - 1], shape[i]);
    blankweight.randomize();
    weights.push_back(blankweight);
  }
}
nntype neural_net::ReLU(nntype x){
  if(x >= 0) return x; else return 0;
}
vector<nntype> neural_net::activation(vector<nntype> input){
  vector<nntype> current = input;
  for(int i = 0; i < weights.size(); i++){
    current = weights[i].timesV(current);
    for(int j = 0; j < current.size(); j++){
      current[j] = ReLU(current[j]);
    }
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

nntype neural_net::partial_derivative_num
(vector<nntype> x, vector<nntype> target, vector<int> mrc){
  // mrc stands for matrix row column
  nntype error = error_datum(x, target);
  weights[mrc[0]].set_element
    (mrc[1], mrc[2], weights[mrc[0]].e(mrc[1], mrc[2]) + differential);
  nntype error_plus_h = error_datum(x, target);
  weights[mrc[0]].set_element
    (mrc[1], mrc[2], weights[mrc[0]].e(mrc[1], mrc[2]) - differential);
  return (error_plus_h - error) / differential;
}
void neural_net::learn(vector<nntype> x, vector<nntype> target){
  vector<int> coord = {0, 0, 0};
  for(int i = 0; i < weights.size(); i++){
    coord[0] = i;
    for(int j = 0; j < weights[i].rows; j++){
      coord[1] = j;
      for(int k = 0; k < weights[i].columns; k++){
        coord[2] = k;
        nntype partial = partial_derivative_num(x, target, coord);
        weights[i].set_element(j, k, weights[i].e(j, k) - (step_size * partial));
      }
    }
  }
}
