#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include "log_reg.h"
using namespace std;


vector<vector<double> > input;		/* input data */
vector<int> target;				/* target classes */

int P, N, K;

//int mean[20], var[20];
int conf_matrix[10][10];

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
				//mean[j] += x;
			}
		}

		// for (int j = 0; j < P; ++j) {
		// 	mean[j] /= N;
		// }

		// for (int i = 0; i < N; ++i) {
		// 	for (int j = 0; j < P; ++j) {
		// 		double r = input[i][j+1] - mean[j];
		// 		var[j] += r * r * (1.0 / N);
		// 	}
		// }

		// for (int i = 0; i < N; ++i) {
		// 	for (int j = 0; j < P; ++j) {
		// 		input[i][j+1] = (input[i][j+1] - mean[j]) / var[j];
		// 	}
		// }
	}
	// read target data
	{
		int c;
		for(int i = 0; i < N; ++i) {
			scanf("%d",&c);
			target.push_back(c-1);
		}
	}

	LogisticLearner learner(K, input, target);
	learner.setConvergenceMargin(1e-3);
	learner.setLearningRate(1);
	//learner.setIterationLimit(12);
	learner.train();
	learner.printParameters();

	LogisticRegressor regressor = learner.getRegressor();
	for (int i = 0; i < N; ++i) {
		int label = regressor.classify(input[i]);
		conf_matrix[target[i]][label]++;
	}
	int correct = 0;
	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j) {
			printf("%d ", conf_matrix[i][j]);
		}
		printf("\n");
		correct += conf_matrix[i][i];
	}
	double success = 1.0 * correct/N * 100;
	printf("success rate: %.3lf%%, error rate: %.3lf%%\n", success, 100 - success);
	return 0;
}