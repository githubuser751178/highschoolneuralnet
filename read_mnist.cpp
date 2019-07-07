#include "neuralnet.hpp"
#include <sstream>
#include <fstream>

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
	while (getline (myfile, line)){
		int v_count = 0;
		count += 1;
		if (count > size) break;
		stringstream ss (line);
		while (ss.good ()){
			getline (ss, sval, ',');
			double val = stod(sval);
			if (v_count == 0)
				label = val;
			else if(val > 0)
				pixels[v_count - 1] = (val / 255.0) - .50;
			v_count += 1;
		}
		train_img img = train_img(label, pixels);
		img_list.push_back(img);
	}
	myfile.close();
	return img_list;
}
