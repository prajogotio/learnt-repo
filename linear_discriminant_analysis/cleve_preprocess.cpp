#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <cmath>
using namespace std;

vector<vector<double> > x;
vector<int> t;
vector<int> idx[2];
double mean[15], var[15];

int main(){
	ifstream data("processed.cleveland.data");
	string line;
	bool ok = true;
	while(getline(data, line)) {
		if(ok) {
			x.push_back(vector<double>(13));
			t.push_back(0);
		}
		ok = true;
		istringstream linestream(line);
		string val;
		int cnt = 0;
		int k = 0;
		while(getline(linestream, val, ',')) {
			if(val == "?") {
				ok = false;
				break;
			}
			istringstream s(val);
			if(cnt < 13) {
				double cur;
				s >> cur;
				if(cnt == 1 || cnt == 2 || cnt==5||cnt == 6 || cnt == 8 || cnt == 10 || cnt == 12) {

				} else {
					x.back()[k] = cur;
					mean[k] += cur;
					k++;
				}
			} else {
				int label;
				s >> label;
				int which = (label == 0 ? 0 : 1);
				t.back() = which+1;
				idx[which].push_back(t.size()-1);
			}
			++cnt;
		}
	}
	for(int k=0;k<6;++k){
		mean[k] /= (double) t.size();
	}
	for(int i=0;i<t.size();++i){
		for(int k=0;k<6;++k){
			double temp = x[i][k] - mean[k];
			var[k] += temp * temp / ((double) t.size());
		}
	}
	for(int i=0;i<t.size();++i){
		for(int k=0;k<6;++k){
			x[i][k] = (x[i][k] - mean[k])/sqrt(var[k]);
		}
	}
	random_device rd;
	mt19937 g(rd());
	shuffle(idx[0].begin(), idx[0].end(), g);
	shuffle(idx[1].begin(), idx[1].end(), g);
	double ratio = 0.85;
	int n[2] = {idx[0].size() * ratio, idx[1].size() * ratio};
	ofstream training("cleveland-training.in");
	ofstream testing("cleveland-testing.in");
	training << "2 6 " << (n[0]+n[1]) << endl;
	for (int i = 0; i < n[0]; ++i) {
		for (int j=0;j<6;++j){
			training << x[idx[0][i]][j] << " ";
		}
		training << endl;
	}
	for (int i = 0; i < n[1]; ++i) {
		for (int j=0;j<6;++j){
			training << x[idx[1][i]][j] << " ";
		}
		training << endl;
	}
	for (int i = 0; i < n[0]; ++i) {
		training << "1" << endl;
		// training << t[idx[0][i]] << endl;
	}
	for (int i = 0; i < n[1]; ++i) {
		training << "2" << endl;
		// training << t[idx[1][i]] << endl;
	}

	testing << "2 6 " << ((int)idx[0].size() + idx[1].size() - n[0] - n[1]) << endl;
	for (int i = n[0]; i < idx[0].size(); ++i) {
		for (int j=0;j<6;++j){
			testing << x[idx[0][i]][j] << " ";
		}
		testing << endl;
	}
	for (int i = n[1]; i < idx[1].size(); ++i) {
		for (int j=0;j<6;++j){
			testing << x[idx[1][i]][j] << " ";
		}
		testing << endl;
	}
	for (int i = n[0]; i < idx[0].size(); ++i) {
		testing << "1" << endl;
		// testing << t[idx[0][i]] << endl;
	}
	for (int i = n[1]; i < idx[1].size(); ++i) {
		testing << "2" << endl;
		// testing << t[idx[1][i]] << endl;
	}

	return 0;
}