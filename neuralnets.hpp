#include <iostream>
#include <vector>
#include <random>
using namespace std;
class matrix {
	public:
		int rows, columns;
		//vector< <vector<double> > m;
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
		vector<double> activation(vector<double>);
		double error_datum (vector<double>, vector<double>);
		double error_data (vector< vector<double> >, vector< vector<double> >);
		double error_derivative_numeric
		(vector<double>, vector<double>, vector<int>, double);
};
