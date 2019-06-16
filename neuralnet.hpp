#include <iostream>
#include <vector>
#include <random>
using namespace std;

typedef double nntype;
typedef vector<nntype> nntype_vector;

class matrix {
	public:
		int rows, columns;
		vector< vector<nntype> > m;
		matrix (int,int);
		nntype sum_elements();
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

class randp {
	public:
		randp(int);
		vector<int> nums;
		int nums_left;
		int next_int();
};
/*
class nn_map {
	public:
		vector<pair<vector<nntype>, vector<nntype> > > map;
		void insert(vector<nntype>, vector<nntype>);
		vector<nntype> at(vector<nntype>);
		bool contains(vector<nntype>);
};

struct vector_hash{
    int operator()(const nntype_vector &V) const {
        int hash=0;
        for(int i=0;i<V.size();i++) {
            hash+=V[i]; // Can be anything
        }
        return hash;
    }
};
*/

class neural_net {
	public:
		vector<int> shape;
		vector<nntype> between_layers;
		int inputs;
		bool weights_changed;
		vector<matrix> weights;
		vector<matrix> corrections;
		nntype step_size;
		nntype differential;
		neural_net (vector<int> , int, nntype, nntype);
		//nn_map activationMemo;
		vector<nntype> activation(vector<nntype>);
		nntype ReLU (nntype);
		nntype error_datum (vector<nntype>, vector<nntype>);
		nntype error_data (vector< vector<nntype> >, vector< vector<nntype> >);
		nntype partial_derivative_num (vector<nntype>, vector<nntype>, nntype &);
		nntype partial_derivative (const vector<nntype>&, const vector<nntype>&, const vector<nntype>&, int, int, int); 
		void learn (const vector<nntype>&, const vector<nntype>&, const vector<nntype>&);
		int get_digit(vector<nntype>);
		vector<nntype> get_vector(int);
		int identify (vector<nntype>);
		void train (vector<train_img>);
		nntype test (vector<train_img>);
		//unordered_map<nntype_vector, nntype_vector, container_hash<nntype_vector>> map;
		//unordered_map <nntype_vector, nntype_vector, vector_hash> memo;
};


vector<train_img> read_mnist(string, int);
vector<xor_input> get_inputs(int);
nntype dot(vector<nntype>, vector<nntype>);
nntype percent_error(nntype, nntype);
