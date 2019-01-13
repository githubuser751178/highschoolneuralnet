#include <iostream>
#include <vector>
#include <random>
using namespace std;

typedef double nntype;

class matrix {
	public:
		int rows, columns;
		//vector< <vector<nntype> > m;
		vector< vector<double> > m;
		matrix (int,int);
		void set_element (int,int,double);
		void randomize ();
		vector<double> timesV(vector<double>);
		vector<double> get_column(int);
		double e(int,int);
		matrix times(matrix);
		matrix plus(matrix);
		matrix scalar_times(double);
};
class neural_net {
	public:
		vector<int> shape;
		int inputs;
		vector<matrix> weights;
		neural_net (vector<int> , int);
		vector<nntype> activation(vector<nntype>);
		nntype error_datum (vector<nntype>, vector<nntype>);
		nntype error_data (vector< vector<nntype> >, vector< vector<nntype> >);
		nntype partial_derivative_num
		(vector<nntype>, vector<nntype>, vector<int>, nntype);
};
