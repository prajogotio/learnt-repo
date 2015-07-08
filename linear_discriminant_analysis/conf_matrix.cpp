#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
using namespace std;


vector<vector<double> > input;		/* input data */
vector<int> target;				/* target classes */

vector<vector<double> > mean;		/* mean of classes */
vector<double> prior;				/* prior of classes */
vector<vector<double> > cov;		/* covariance matrix */
vector<vector<double> > inv_cov;
vector<vector<double> > coeff;
int P, N, K;

vector<vector<int> > conf_matrix;
double eps = 1e-12;


void computeInverseCovariance() {
	printf("covariance:\n");
	for(int i=0;i<P;++i){
		for(int j=0;j<P;++j){
			printf("%.3lf ", cov[i][j]);
		}
		printf("\n");
	}
	inv_cov = vector<vector<double> >(P, vector<double>(P));
	for(int i = 0; i < P; ++i) {
		inv_cov[i][i] = 1;
	}
	for(int i=0;i<P;++i){

		printf("debug:\n");
		for (int i =0;i < P; ++i) {
			for(int j = 0; j < P; ++j){
				printf("%.3lf ", cov[i][j]);
			}
			printf("|");
			for (int j =0; j < P; ++ j) {
				printf("%.3lf ", inv_cov[i][j]);
			}
			printf("\n");
		}

		if(abs(cov[i][i]) < eps) {
			for(int j=i+1;j<P;++j){
				if(abs(cov[j][i]) > eps) {
					for(int k=0;k<P;++k){
						cov[i][k] += cov[j][k];
						inv_cov[i][k] += inv_cov[j][k];
					}
					break;
				} 
			}
		}
		double d = cov[i][i];
		for(int k=0;k<P;++k){
			cov[i][k] /= d;
			inv_cov[i][k] /= d;
		}
		for(int j=0;j<P;++j){
			if(j==i) continue;
			double r = cov[j][i];
			for(int k=0;k<P;++k){
				cov[j][k] -= r * cov[i][k];
				inv_cov[j][k] -= r * inv_cov[i][k];
			}
		}
	}
	
	printf("inverse covariance:\n");
	for(int i=0;i<P;++i){
		for(int j=0;j<P;++j){
			printf("%.3lf ", inv_cov[i][j]);
		}
		printf("\n");
	}
}


void computeCoeff() {
	coeff = vector<vector<double> > (K+1, vector<double>(P));
	for (int k=1;k<=K;++k){
		for(int i=0;i<P;++i){
			for(int j=0;j<P;++j){
				coeff[k][i] += inv_cov[i][j] * mean[k][j];
			}
		}
	}
}



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

	cin >> K >> P;
	// prior
	prior = vector<double>(K+1);
	for (int i = 1; i <= K; ++i) {
		cin >> prior[i];
	}

	// mean
	mean = vector<vector<double> >(K+1, vector<double>(P));
	for (int i = 1; i <= K; ++i) {
		for(int j = 0; j < P; ++j) {
			cin >> mean[i][j];
		}
	}

	// covariance matrix
	cov = vector<vector<double> >(P, vector<double>(P));
	for (int i = 0; i < P; ++i){
		for(int j = 0; j < P; ++j){
			cin >> cov[i][j];
		}
	}
	
	computeInverseCovariance();
	computeCoeff();

	// confusion matrix
	conf_matrix = vector<vector<int> >(K+1, vector<int>(K+1));
	for(int i = 0; i < N; ++i) {
		double hi = -1e308;
		int label = 0;
		for (int k = 1; k <= K; ++k) {
			double lin = 0, cons = 0;
			for(int j=0;j<P;++j){
				lin += input[i][j] * coeff[k][j];
				cons += mean[k][j] * coeff[k][j];
			}
			double delta = log(prior[k]) + lin - 0.5 * cons;
			if (delta > hi ) {
				label = k;
				hi = delta;
			}
		}
		conf_matrix[target[i]][label]++;
	}
	double correct = 0;
	for(int i=1;i<=K;++i){
		correct += conf_matrix[i][i];
		for(int j=1;j<=K;++j){
			printf("%d ", conf_matrix[i][j]);
		}
		printf("\n");
	}
	printf("success rate: %.2lf, error rate: %.2lf\n", correct/N*100, (1-correct/N)*100);
	return 0;
}