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
		 
vector<train_img> random_read(string filename, int file_size, int num_img){
	ifstream my_file(filename);
	randp rng(file_size);
	string line, str_csv_val;
	int label;
	int img_index[num_img];
	int line_count = 0;
	vector<nntype> pixels;
	vector<train_img> img_list;
	for(int i = 0; i < num_img; i++){
		img_index[i] = rng.next_int();
	}
	while(getline(my_file, line)){
		bool label_read = false;
		if(img_list.size() == num_img)
			break;
		else if(line_count == img_index[img_list.size()]){
			stringstream ss (line);
			while(ss.good()){
				getline(ss, str_csv_val, ','); 
				nntype csv_val = stod(str_csv_val);
				if(!label_read){
					label = csv_val;
					label = true;
				} else {
					pixels.push_back(csv_val / 255);
				}
			}
			train_img img(label, pixels);
			img_list.push_back(img);
		}
	}
	my_file.close();
	return img_list;
}


//I'm trying to read in random lines from this csv
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
	//cout << "file read finished" << endl;
	return img_list;
}
