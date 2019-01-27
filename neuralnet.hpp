#include <iostream>
#include <vector>
#include <random>
using namespace std;

typedef double nntype;
typedef unsigned char uchar;

class matrix {
	public:
		int rows, columns;
		//vector< <vector<nntype> > m;
		vector< vector<nntype> > m;
		matrix (int,int);
		void set_element (int,int,nntype);
		void randomize ();
		void zero ();
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

class neural_net {
	public:
		vector<int> shape;
		int inputs;
		vector<matrix> weights;
		vector<matrix> corrections;
		nntype step_size;
		nntype differential;
		neural_net (vector<int> , int, nntype, nntype);
		vector<nntype> activation(vector<nntype>);
		nntype ReLU (nntype);
		nntype error_datum (vector<nntype>, vector<nntype>);
		nntype error_data (vector< vector<nntype> >, vector< vector<nntype> >);
		nntype partial_derivative_num (vector<nntype>, vector<nntype>, nntype &);
		void learn (vector<nntype>, vector<nntype>);
		int vectordigit (vector<nntype>);
		vector<nntype> digitvector (nntype digit);
		int identify (vector<nntype>);
		void train (vector<train_img>);
		nntype test (vector<train_img>);
};
vector<train_img> read_mnist(string);
