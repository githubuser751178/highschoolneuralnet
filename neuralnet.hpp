#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include <vector>
#include <cmath>
#include <assert.h>

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
		void print(int, int);
		vector<nntype> timesV(vector<nntype>);
		vector<nntype> get_column(int);
		nntype e(int,int);
		matrix times(matrix);
		matrix plus(matrix);
		matrix capped_plus(matrix, nntype);
		matrix scalar_times(nntype);
		int num_elements();
};
class train_img {
	public:
		int label;
		vector<nntype> pixels;
		train_img (int, vector<nntype>);
};

class randp {
	public:
		randp(int);
		vector<int> nums;
		int nums_left;
		int next_int();
};

class neural_net {
	public:
		neural_net(int, nntype, nntype);

		//member variables
		vector<matrix> weights;
		vector<matrix> corrections;	
		nntype step_size;
		nntype differential;
		bool weights_changed;
		vector<nntype> layer_1_output;
		vector<nntype> layer_2_output;
		
		//helper functions
		nntype logistic(nntype);
		nntype dlogistic(nntype);

		int get_digit(vector<nntype>);
		vector<nntype> get_vector(int);

		nntype error_datum(vector<nntype>, vector<nntype>);
		nntype error_data(vector< vector<nntype> >, vector< vector<nntype> >);

		nntype partial_derivative_num (vector<nntype>, vector<nntype>, nntype &);
		nntype partial_derivative(const vector<nntype>&, const vector<nntype>&, const vector<nntype>&, int, int, int);
		int identify (vector<nntype>);
		nntype avg_error(vector<train_img>);

		//neural net functions
		vector<nntype> activation(vector<nntype>);
		void learn (const vector<nntype>&, const vector<nntype>&, const vector<nntype>&);
		void train (vector<train_img>);
		nntype test (vector<train_img>);
};

nntype dot(vector<nntype>, vector<nntype>);
vector<train_img> read_mnist(string, int);
vector<train_img> random_read(string, int, int);
nntype percent_error(nntype, nntype);
