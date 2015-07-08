#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
using namespace std;


vector<vector<double> > input;		/* input data */
vector<int> target;				/* target classes */

vector<vector<double> > mean;		/* mean of classes */
vector<double> prior;				/* prior of classes */
vector<vector<double> > cov;		/* covariance matrix */
int P, N, K;

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
			target.push_back(c);
		}
	}

	// fitting the LDA
	{
		// maximum likelihood estimate of prior
		vector<int> cnt(K+1);
		for(int i = 0; i < N; ++i) {
			cnt[target[i]]++;
		}
		prior.push_back(0);
		for(int k = 1; k <= K; ++k) {
			prior.push_back((double)cnt[k]/N);
		}

		// maximum likelihood estimate of mean
		mean = vector<vector<double> > (K+1, vector<double>(P));
		for(int i = 0; i < N; ++i) {
			for(int j = 0; j < P; ++j) {
				mean[target[i]][j] += input[i][j];
			}
		}
		for(int k = 1; k <= K; ++k) {
			for(int j = 0; j < P; ++j) {
				mean[k][j] /= cnt[k];
			}
		}

		// maximum likelihood estimate of covariance matrix
		cov = vector<vector<double> > (P, vector<double>(P));
		for(int i = 0; i < N; ++i) {
			vector<double> temp(P);
			for(int j = 0; j < P; ++j) {
				temp[j] = input[i][j] - mean[target[i]][j];
			}
			for(int j = 0; j < P; ++j) {
				for(int k = 0; k < P; ++k) {
					cov[j][k] += temp[j] * temp[k] * (1.0 / (N-K));
				}
			}
		}

	}
	// LDA parameters output
	printf("%d %d\n", K, P);

	// prior
	for (int i = 1; i <= K; ++i) {
		printf("%.12lf ", prior[i]);
	}
	printf("\n");

	// mean
	for (int i = 1; i <= K; ++i) {
		for(int j = 0; j < P; ++j) {
			printf("%.12lf ", mean[i][j]);
		}
	}
	printf("\n");

	// covariance matrix
	for (int i = 0; i < P; ++i){
		for(int j = 0; j < P; ++j){
			printf("%.12lf ", cov[i][j]);
		}
		printf("\n");
	}

	return 0;
}