#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include <random>

using namespace Eigen;
using namespace std;

#define double long double
#define pb push_back
#define Matrix MatrixXd

Matrix mulDiag(Matrix M, vector<double> D){
    for(int i=0;i<M.rows();i++)
        for(int j=0;j<M.cols();j++)
            M(i,j)*=D[j];
    return M;
}

Matrix Diagmul(vector<double> D, Matrix M){
    for(int i=0;i<M.rows();i++)
        for(int j=0;j<M.cols();j++)
            M(i,j)*=D[i];
    return M;
}

double norm(Matrix vec){
    return sqrt(vec.squaredNorm());
}
