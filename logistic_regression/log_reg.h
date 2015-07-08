#ifndef _H_LOGISTIC_REGRESSION
#define _H_LOGISTIC_REGRESSION

#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

namespace MachineLearningImplementation {

using namespace std;


class LogisticComputor {
public:
	double dotProduct(const vector<double>& u, const vector<double>& v) {
		double ret = 0;
		for (int i = 0; i < u.size(); ++i) {
			ret += u[i] * v[i];
		}
		return ret;
	}

	double computeProbability(int k, const vector<double>& x, const vector<vector<double> >& b) {
		return numericallyStableSoftmax(k, x, b);
	}
	
private:
	double numericallyStableSoftmax(int k, const vector<double>& x, const vector<vector<double> >& b) {
		vector<double> temp;
		double maxval = 0;
		for (int i = 0; i < b.size(); ++i) {
			temp.push_back(dotProduct(x, b[i]));
			maxval = max(temp.back(), maxval);
		}
		double den = 0;
		for (int i = 0; i < b.size(); ++i) {
			if(temp[i] - maxval < -1e12) continue;
			den += exp(temp[i] - maxval);
		}
		return exp(temp[k] - maxval) / den;
	}
};

class LogisticRegressor {
public:
	LogisticRegressor(int input_K, const vector<vector<double> >& input_b):
		K(input_K),
		b(input_b),
		P(input_b.back().size()),
		computor()
		{}
	~LogisticRegressor() {}

	int classify(const vector<double>& x) {
		double hi = -1e300;
		int label = -1;
		for (int k = 0; k < K; ++k) {
			double cur = computor.computeProbability(k, x, b);
			if (cur > hi) {
				hi = cur;
				label = k;
			}
		}
		
		return label;
	}
	
private:
	vector<vector<double> > b;			/* betas */
	int K;								/* number of classes */
	int P;								/* parameter length */
	LogisticComputor computor;
};

class LogisticLearner {

public:
	LogisticLearner(int input_K, const vector<vector<double> >& input_x, const vector<int>& input_t): 
		K(input_K),
		N(input_x.size()),
		P(input_x.back().size()),
		x(input_x),
		t(input_t),
		b(K, vector<double>(P, 0)),
		db(K, vector<double>(P, 0)),
		H(K, vector<vector<vector<double> > >(K, vector<vector<double> >(P, vector<double>(P, 0)))),
		inv_H(K, vector<vector<vector<double> > >(K, vector<vector<double> >(P, vector<double>(P, 0)))),
		conv_margin(0.001),
		learningRate(1.0),
		tb(K, vector<double>(P, 0)),
		computor(),
		iterationLimit(1000)
		{}

	~LogisticLearner() {}

	void setConvergenceMargin(double conv_margin) {
		if (conv_margin < 0) return;
		this -> conv_margin = conv_margin;
	}

	void setLearningRate(double learningRate) {
		if (learningRate < 0) return;
		this -> learningRate = learningRate;
	}

	void setIterationLimit(int limit) {
		iterationLimit = limit;
	}

	void train() {
		runNewtonRaphsonAlgorithm();
	}

	void printParameters() {
		printf("%d %d\n", K, P);
		for (int k=0;k<K-1;++k){
			for(int i=0;i<P;++i){
				printf("%.24lf ", b[k][i]);
			}
			printf("\n");
		}
	}

	LogisticRegressor getRegressor() {
		return LogisticRegressor(K, b);
	}
private:
	int K;											/* number of classes */
	int N;											/* number of data */
	int P;											/* parameter of x */
	vector<vector<double> > x;						/* input x */
	vector<int> t;									/* class of x, coded to 0, 1, ..., K-1 */
	vector<vector<double> > b;						/* beta */
	vector<vector<double> > db;						/* partial derivative of beta */
	vector< vector<vector<vector<double> > > > H;	/* hessian matrix of beta */
	vector< vector<vector<vector<double> > > > inv_H; /* inverse hessian */
	double conv_margin;								/* convergence margin */
	double learningRate;	
	int iterationLimit;		
	static const double EPS;

	vector<vector<double> > tb;						/* temporary storage */
	LogisticComputor computor;

	double dotProduct(const vector<double>& u, const vector<double>& v) {
		return computor.dotProduct(u, v);
	}

	double computeProbability(int k, const vector<double>& x) {
		return computor.computeProbability(k, x, b);
	}


	double computeLikelihood() {
		double ret = 0;
		for (int i = 0; i < N; ++i) {
			ret += log(computeProbability(t[i], x[i]));
		}
		return ret;
	}

	void computePartialDerivatives() {
		for (int k = 0; k < K-1; ++k) {
			for(int i = 0; i < P; ++i) db[k][i] = 0;
			for(int i = 0; i < N; ++i) {
				double ratio = ((t[i] == k ? 1.0 : 0.0) - computeProbability(k, x[i]));
				for (int j = 0; j < P; ++j) {
					db[k][j] += ratio * x[i][j];
				}
			}
		}
		// printf("partial derivatives:\n");
		// for (int k = 0; k < K-1; ++k) {
		// 	for (int i = 0; i < P; ++i) {
		// 		printf("%.3lf ", db[k][i]);
		// 	}
		// 	printf("\n");
		// }
	}


	void computeHessians() {
		for (int j = 0; j < K-1; ++j) {
			for (int k = 0; k < K-1; ++k) {
				for (int q = 0; q < P; ++q) {
					for (int p = 0; p < P; ++p) {
						H[j][k][q][p] = 0;
					}
				}
			}
		}
		for (int j = 0; j < K-1; ++j) {
			for (int k = 0; k < K-1; ++k) {
				for (int i = 0; i < N; ++i) {
					for (int q = 0; q < P; ++q) {
						for (int p = 0; p < P; ++p) {
							double pr = computeProbability(k, x[i]);
							double qr = (j == k ? 1 : 0) - computeProbability(j, x[i]);
							H[j][k][q][p] += - x[i][q] * x[i][p] * pr * qr;
						}
					}
				}
			}
		}
		// printf("hessian:\n");
		// for(int j=0;j<K-1;++j){
		// 	for(int p=0;p<P;++p){
		// 		for(int k=0;k<K-1;++k){
		// 			for(int q=0;q<P;++q){
		// 				printf("%.3lf ", H[j][k][p][q]);
		// 			}
		// 		}
		// 		printf("\n");
		// 	}
		// }
	}

	void computeInverseHessians() {
		for (int j = 0; j < K-1; ++j) {
			for (int k = 0; k < K-1; ++k) {
				for(int q = 0; q < P; ++q) {
					for (int p = 0; p < P; ++p) {
						inv_H[j][k][q][p] = 0;
						if (j==k && q==p) inv_H[j][k][q][p] = 1;
					}
				}
			}
		}
		for (int j = 0; j < K-1; ++j) {
			for (int q = 0; q < P; ++q) {
				if (abs(H[j][j][q][q]) < EPS) {
					bool done = false;
					for (int k = j; k < K-1 && !done; ++k) {
						for (int p = q+1; p < P && !done; ++p) {
							if (abs(H[k][j][p][q]) < EPS) continue;
							for (int u = 0; u < K-1; ++u) {
								for (int v = 0; v < P; ++v) {
									H[j][u][q][v] += H[k][u][p][v];
									inv_H[j][u][q][v] += inv_H[k][u][p][v];
								}
							}
							done = true;
						}
					}
				}
				double den = H[j][j][q][q];
				for (int k = 0;k<K-1;++k){
					for(int p=0;p<P;++p){
						H[j][k][q][p] /= den;
						inv_H[j][k][q][p] /= den;
					}
				}
				for (int k = 0; k<K-1;++k){
					for(int p=0;p<P;++p){
						if(j==k && q==p) continue;
						double mult = H[k][j][p][q];
						for (int u = 0; u < K-1; ++u) {
							for (int v = 0; v < P; ++v) {
								H[k][u][p][v] -= mult * H[j][u][q][v];
								inv_H[k][u][p][v] -= mult * inv_H[j][u][q][v];
							}
						}
					}
				}
			}
		}
		// printf("inverse hessian:\n");
		// for(int j=0;j<K-1;++j){
		// 	for(int p=0;p<P;++p){
		// 		for(int k=0;k<K-1;++k){
		// 			for(int q=0;q<P;++q){
		// 				printf("%.3lf ", inv_H[j][k][p][q]);
		// 			}
		// 		}
		// 		printf("\n");
		// 	}
		// }
	}

	void runNewtonRaphsonAlgorithm() {
		double prevLikelihood = computeLikelihood();
		printf("starting likelihood: %.6lf\n", prevLikelihood);
		int cnt = 0;
		while (1) {
			computePartialDerivatives();
			computeHessians();
			computeInverseHessians();
			for (int k = 0; k < K-1; ++k) {
				for (int i = 0; i < P; ++i) {
					tb[k][i] = 0;
				}
			}
			for (int j = 0; j < K-1; ++j) {
				for (int k = 0; k < K-1; ++k) {
					for (int q = 0; q < P; ++q) {
						for (int p = 0; p < P; ++p) {
							tb[j][q] += inv_H[j][k][q][p] * db[k][p];
						}
					}
				}
			}
			for (int k = 0; k < K-1; ++k) {
				for (int i = 0; i < P; ++i) {
					b[k][i] = b[k][i] - learningRate * tb[k][i];
				}
			}
			// printf("betas:\n");
			// for (int k = 0; k < K; ++k) {
			// 	for(int i = 0; i < P; ++i) printf("%.3lf ", b[k][i]);
			// 	printf("\n");
			// }
			double currentLikelihood = computeLikelihood();
			printf("iteration: %d likelihood: %.6lf\n", cnt++, currentLikelihood);
			if (abs(currentLikelihood - prevLikelihood) < conv_margin) {
				break;
			}
			prevLikelihood = currentLikelihood;
			if(cnt == iterationLimit) break;
		}
	}
};

const double LogisticLearner::EPS = 1e-12;


}

typedef MachineLearningImplementation::LogisticRegressor LogisticRegressor;
typedef MachineLearningImplementation::LogisticLearner LogisticLearner;
#endif