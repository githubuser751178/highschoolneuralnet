#include <iostream>
#include <vector>
#include <random>
#include "neuralnets.hpp"
using namespace std;

neural_net::neural_net(vector<int> s, int i){
  shape = s;
  inputs = i;
  matrix blankweight1 (inputs,shape[0]);
  blankweight1.randomize();
  weights.push_back(blankweight1);
  for(int i=1; i<shape.size(); i++){
    matrix blankweight (shape[i-1],shape[i]);
    blankweight.randomize();
    weights.push_back(blankweight);
  }
}
vector<double> neural_net::activation(vector<double> input){
  vector<double> current = input;
  for(int i=0; i<weights.size(); i++){
    current = weights[i].timesV(current);
  }
  return current;
}
double neural_net::error_datum(vector<double> input, vector<double> target){
  double error = 0;
  vector<double> output = activation(input);
  for(int i=0; i<target.size(); i++){
    error += pow(target[i]-output[i],2);
  }
  return error;
}
double neural_net::error_data
(vector< vector<double> > inputs, vector< vector<double> > targets){
  double total_e = 0;
  for(int i=0; i<inputs.size(); i++)
    total_e += error_datum(inputs[i],targets[i]);
  return total_e;
}

double neural_net::error_derivative_numeric
(vector<double> x, vector<double> target, vector<int> w_coord, double h){
  matrix m = weights[w_coord[0]];
  vector<matrix> weights_plus_h = weights;
  weights_plus_h[w_coord[0]].set_element(w_coord[1],w_coord[2],m.e(w_coord[1],w_coord[2])+h);
  neural_net plus_h (shape,inputs);
  plus_h.weights = weights_plus_h;
  return (plus_h.error_datum(x,target)-error_datum(x,target))/h;
}
