#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "neuralnet.hpp"

using namespace std;

typedef double nntype;
train_img::train_img(int l, vector<nntype> p){
	label = l;
	pixels = p;
}
vector<train_img> read_mnist (string filename, int size){
	ifstream myfile (filename);
	int label;
	int count = 0;
	string line, sval;
	vector<train_img> img_list;
	vector<nntype> pixels;
	for (int i = 0; i < 784; i++)
		pixels.push_back(0);
	//cout << "pixels initiated" << endl;
	while (getline (myfile, line)){
		int v_count = 0;
		count += 1;
		if (count > size) break;
		stringstream ss (line);
		while (ss.good ()){
			getline (ss, sval, ',');
			int val = stoi(sval);
			if (v_count == 0)
				label = val;
			else
				pixels[v_count - 1] = val;
			v_count += 1;
		}
		train_img img = train_img(label, pixels);
		img_list.push_back(img);
	}
	myfile.close();
	//cout << "file read finished" << endl;
	return img_list;
}
