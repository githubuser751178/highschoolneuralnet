#include <iostream>
#include <vector>
#include <random>
#include "neuralnet.hpp"

using namespace std;
typedef double nntype;

matrix::matrix(int r, int c){
	rows = r;
	columns = c;
	vector<nntype> dummy;
	for(int i=0; i<columns; i++)
		dummy.push_back(0);
	for(int i=0; i<rows; i++)
		m.push_back(dummy);
}
nntype matrix::e(int i, int j){
	return m[i][j];
}
void matrix::set_element (int r, int c, nntype e){
	m[r][c] = e;
}
void matrix::print(){
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < columns; j++){
			cout << m[i][j] << " ";
		}
		cout << "\n";
	}
}
void matrix::randomize (){
	for(int i=0; i<rows; i++){
		for(int j = 0; j < columns; j++){
			nntype x = (((nntype)rand()/(nntype)RAND_MAX)-.5);
			m[i][j] = x;
		}
	}
}
void matrix::zero (){
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < columns; j++){
			m[i][j] = 0;
		}
	}
}

nntype dot(vector<nntype> v1, vector<nntype> v2){
	nntype sum = 0;
	if(v1.size() == v2.size()){
		for(int i=0; i<v1.size(); i++)
			sum += v1[i]*v2[i];
		return sum;
	} else{
		return 0;
	}
}
vector<nntype> matrix::timesV(vector<nntype> v){
	vector<nntype> product;
	for(int i=0; i<m.size(); i++){
		product.push_back(dot(m[i],v));
	}
	return product;
}
vector<nntype> matrix::get_column(int j){
	vector<nntype> col;
	for(int i=0; i<rows; i++){
		col.push_back(m[i][j]);
	}
	return col;
}
matrix matrix::times(matrix m2){
	matrix product (rows,m2.columns);
	for(int i=0; i<rows; i++){
		for(int j=0; j<m2.columns; j++){
			vector<nntype> col;
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
matrix matrix::scalar_times(nntype c){
	matrix product(rows,columns);
	for(int i=0; i<rows; i++){
		for(int j=0; j<columns; j++)
			product.set_element(i,j,c*m[i][j]);
	}
	return product;
}
