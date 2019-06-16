#include <vector>
#include "neuralnet.hpp"
#include <random>
#include <time.h>

using namespace std;

randp::randp(int n){
	for(int i = 0; i < n; i++){
		nums.push_back(i);
	}
}

int randp::next_int(){
	if(nums.size() == 0)
		return -1;
	srand(time(NULL));
	int index = (rand() % nums.size());
	int num = nums[index];
	nums.erase(nums.begin() + index);
	return num;
}
