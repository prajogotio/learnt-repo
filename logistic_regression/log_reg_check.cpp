#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include "log_reg.h"
using namespace std;


vector<vector<double> > input;		/* input data */
vector<int> target;				/* target classes */
vector<vector<double> > b;
int P, N, K;

vector<vector<int> > conf_matrix;
double eps = 1e-12;




int main(){
	// get the number of classes
	scanf("%d",&K);
	// get the dimension of input x
	scanf("%d",&P);
	// get number of training data
	scanf("%d",&N);

	// read training data
	{
		double x;
		for(int i = 0; i < N; ++i) {
			input.push_back(vector<double>());
			input.back().push_back(1);
			for (int j = 0; j < P; ++j) {
				scanf("%lf",&x);
				input.back().push_back(x);
			}
		}
	}
	// read target data
	{
		int c;
		for(int i = 0; i < N; ++i) {
			scanf("%d",&c);
			target.push_back(c-1);
		}
	}

	P++;
	b = vector<vector<double> > (K, vector<double>(P, 0));
	for (int i = 0; i < K-1; ++i){
		for (int j = 0; j < P; ++j){
			cin >> b[i][j];
		}
	}
	LogisticRegressor regressor(K, b);
	conf_matrix = vector<vector<int> > (K, vector<int>(K, 0));
	for (int i = 0; i < N; ++i) {
		int label = regressor.classify(input[i]);
		conf_matrix[target[i]][label]++;
	}

	double correct = 0;
	for(int i=0;i<K;++i){
		correct += conf_matrix[i][i];
		for(int j=0;j<K;++j){
			printf("%d ", conf_matrix[i][j]);
		}
		printf("\n");
	}
	printf("success rate: %.2lf, error rate: %.2lf\n", correct/N*100, (1-correct/N)*100);
	return 0;
}