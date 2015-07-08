#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <fstream>
#include <vector>
using namespace std;

double data[150][4];
int label[150];
vector<int> idx;


int main(){
	string line;
	ofstream training("training.in");
	ofstream test("test.in");
	idx = vector<int>(150);
	for(int i = 0; i < 150; ++i) {
		char c;
		for(int j=0;j<4;++j){
			cin >> data[i][j] >> c;
		}
		cin >> line;
		label[i] = i/50 + 1;
		idx[i] = i;
	}
	random_device rd;
	mt19937 g(rd());
	shuffle(idx.begin(), idx.begin()+50, g);
	shuffle(idx.begin()+50, idx.begin()+100, g);
	shuffle(idx.begin()+100, idx.begin()+150, g);
	training << "3 4 120" << endl;
	test << "3 4 30" << endl;
	for(int i=0;i<3;++i){
		for(int j=0;j<40;++j){
			for(int k=0;k<4;++k){
				training << data[idx[i*50 + j]][k] << " ";
			}
			training << endl;
		}
		for(int j=40;j<50;++j){
			for(int k=0;k<4;++k){
				test << data[idx[i*50 + j]][k] << " ";
			}
			test << endl;
		}
	}
	for(int i=0;i<3;++i){
		for(int j=0;j<40;++j){
			training << label[idx[i*50 + j]] << endl;
		}
		for(int j=40;j<50;++j){
			test << label[idx[i*50 + j]] << endl;
		}
	}
	return 0;
}