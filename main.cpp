#include <iostream>
#include <vector>
#include <random>
#include "neuralnets.hpp"
using namespace std;

int main () {
  matrix m1 (2,2);
	vector< vector<int> > a;
	vector<int> r1;
	r1.push_back(1);
	a.push_back(r1);
	//cout << a[0][0] <<endl;
  m1.set_element (0,0,2);
	m1.set_element (1,1,2);
	matrix m2 (2,2);
	vector<double> v1;
	v1.push_back(1);
	v1.push_back(2);
	//cout << m1.m[0][0] <<endl;
	//cout << m1.m[0][1] <<endl;
	vector<double> v2;
	v2 = m1.timesV(v1);
	cout<<v1[0]<<endl;
	cout<<v1[1]<<endl;
	m2.set_element (0,0,1);
	m2.set_element (0,1,2);
	m2.set_element (1,0,3);
	m2.set_element (1,1,4);
	matrix m3 (2,2);
	m3 = m1.times(m2);
	cout<<m3.e(0,0)<<endl;
	cout<<m3.e(0,1)<<endl;
	cout<<m3.e(1,0)<<endl;
	cout<<m3.e(1,1)<<endl;
  return 0;
}
