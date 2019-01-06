#include <iostream>
#include <vector>
#include <random>
#include "neuralnets.hpp"

using namespace std;

matrix::matrix(int r, int c){
	rows = r;
	columns = c;
	vector<double> dummy;
		for(int i=0; i<columns; i++)
			dummy.push_back(0);
		for(int i=0; i<rows; i++)
			m.push_back(dummy);
}
double matrix::e(int i, int j){
	return m[i][j];
}
void matrix::set_element (int r, int c, double e){
	m[r][c] = e;
}
void matrix::randomize (){
	for(int i=0; i<rows; i++){
		for(int j=0; j<rows; j++){
			double x =(((double)rand()/(double)RAND_MAX)-.5);
			m[i][j] = x;
		}
	}
}
double dot(vector<double> v1, vector<double> v2){
	double sum = 0;
	if(v1.size() == v2.size()){
		for(int i=0; i<v1.size(); i++)
			sum += v1[i]*v2[i];
		return sum;
	} else{
		return 0;
	}
}
vector<double> matrix::timesV(vector<double> v){
	vector<double> product;
	for(int i=0; i<m.size(); i++){
		product.push_back(dot(m[i],v));
	}
	return product;
}
vector<double> matrix::get_column(int j){
	vector<double> col;
	for(int i=0; i<rows; i++){
		col.push_back(m[i][j]);
	}
	return col;
}
matrix matrix::times(matrix m2){
	matrix product (rows,m2.columns);
	for(int i=0; i<rows; i++){
		for(int j=0; j<m2.columns; j++){
			vector<double> col;
			col = m2.get_column(j);
			product.set_element(i,j,dot(m[i],col));
		}
	}
	return product;
}
matrix matrix::plus(matrix m2){
	matrix sum (rows,columns);
	for(int i=0; i<rows; i++){
		for(int j=0; j<columns; j++){
			sum.set_element(i,j,m[i][j]+m2.m[i][j]);
		}
	}
	return sum;
}
matrix matrix::scalar_times(double c){
	matrix product(rows,columns);
	for(int i=0; i<rows; i++){
		for(int j=0; j<columns; j++)
			product.set_element(i,j,c*m[i][j]);
	}
	return product;
}
