#include <iostream>
#include <vector>
#include <random>
#include "neuralnets.hpp"
using namespace std;

typedef double nntype;

neural_net::neural_net(vector<int> s, int i){
  shape = s;
  inputs = i;
  matrix blankweight1 (inputs, shape[0]);
  blankweight1.randomize();
  weights.push_back(blankweight1);
  for(int i = 1; i < shape.size(); i++){
    matrix blankweight (shape[i - 1], shape[i]);
    blankweight.randomize();
    weights.push_back(blankweight);
  }
}
vector<nntype> neural_net::activation(vector<nntype> input){
  vector<nntype> current = input;
  for(int i = 0; i < weights.size(); i++){
    current = weights[i].timesV(current);
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
(vector<nntype> x, vector<nntype> target, vector<int> w_coord, nntype h){
  nntype error = error_datum(x, target);
  weights[w_coord[0]].set_element
    (w_coord[1],w_coord[2],m.e(w_coord[1],w_coord[2]) + h);
  nntype error_plus_h = error_datum(x, target);
  weights[w_coord[0]].set_element
    (w_coord[1],w_coord[2],m.e(w_coord[1],w_coord[2]) - h);
  return (error_plus_h - error) / h;
}
