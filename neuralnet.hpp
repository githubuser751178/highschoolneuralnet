#include <iostream>
#include <unordered_map>
#include <vector>
#include <random>
using namespace std;

typedef double nntype;

class matrix {
	public:
		int rows, columns;
		vector< vector<nntype> > m;
		matrix (int,int);
		void set_element (int,int,nntype);
		void randomize ();
		void zero ();
		void print ();
		vector<nntype> timesV(vector<nntype>);
		vector<nntype> get_column(int);
		nntype e(int,int);
		matrix times(matrix);
		matrix plus(matrix);
		matrix scalar_times(nntype);
};
class train_img {
	public:
		int label;
		vector<nntype> pixels;
		train_img (int, vector<nntype>);
};

class xor_input {
	public:
		int label;
		vector<nntype> target;
		vector<nntype> things;
		xor_input(nntype, nntype);
};

class nn_map {
	public:
		vector<pair<vector<nntype>, vector<nntype> > > map;
		void insert(vector<nntype>, vector<nntype>);
		vector<nntype> at(vector<nntype>);
		bool contains(vector<nntype>);
};

class neural_net {
	public:
		vector<int> shape;
		vector<nntype> between_layers;
		int inputs;
		vector<matrix> weights;
		vector<matrix> corrections;
		nntype step_size;
		nntype differential;
		neural_net (vector<int> , int, nntype, nntype);
		nn_map activationMemo;
		vector<nntype> activation(vector<nntype>);
		nntype ReLU (nntype);
		nntype error_datum (vector<nntype>, vector<nntype>);
		nntype error_data (vector< vector<nntype> >, vector< vector<nntype> >);
		nntype partial_derivative_num (vector<nntype>, vector<nntype>, nntype &);
		nntype partial_derivative (vector<nntype>, vector<nntype>, int, int, int); 
		void learn (vector<nntype>, vector<nntype>);
		int vectordigit (vector<nntype>);
		int identify (vector<nntype>);
		void train (vector<xor_input>);
		nntype test (vector<xor_input>);
};

vector<train_img> read_mnist(string, int);
vector<xor_input> get_inputs(int);
nntype dot(vector<nntype>, vector<nntype>);
