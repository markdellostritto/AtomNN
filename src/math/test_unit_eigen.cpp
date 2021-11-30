//c
#include <ctime>
//c++
#include <iostream>
#include <stdexcept>
//eigen
#include <Eigen/Dense>

double test_eigensolver_values(int b, int e, int s){
	typedef Eigen::MatrixXf MatType;
	double sum=0;
	for(int i=b; i<=e; i+=s){
		clock_t start_=std::clock();
		MatType mat=MatType::Random(i,i);
		Eigen::EigenSolver<MatType> solver(mat);
		sum+=solver.eigenvalues()[0].real();
		clock_t stop_=std::clock();
		const double time=((double)(stop_-start_))/CLOCKS_PER_SEC;
		std::cout<<"time["<<i<<"] = "<<time<<"\n";
	}
	return sum;
}

double test_selfadjointeigensolver_values(int b, int e, int s){
	typedef Eigen::MatrixXf MatType;
	double sum=0;
	for(int i=b; i<=e; i+=s){
		clock_t start_=std::clock();
		MatType mat=MatType::Random(i,i);
		Eigen::SelfAdjointEigenSolver<MatType> solver(mat);
		sum+=solver.eigenvalues()[0];
		clock_t stop_=std::clock();
		const double time=((double)(stop_-start_))/CLOCKS_PER_SEC;
		std::cout<<"time["<<i<<"] = "<<time<<"\n";
	}
	return sum;
}

double test_selfadjointeigensolver_rank(int b, int e, int s){
	typedef Eigen::MatrixXf MatType;
	double sum=0;
	for(int i=b; i<=e; i+=s){
		clock_t start_=std::clock();
		MatType mat=MatType::Random(i,i);
		//Eigen::FullPivLU<MatType> solver(mat);
		Eigen::FullPivHouseholderQR<MatType> solver(mat);
		sum+=solver.rank();
		clock_t stop_=std::clock();
		const double time=((double)(stop_-start_))/CLOCKS_PER_SEC;
		std::cout<<"time["<<i<<"] = "<<time<<"\n";
	}
	return sum;
}

int main(int argc, char* argv[]){
	
	test_eigensolver_values(1,1001,100);
	test_selfadjointeigensolver_values(1,1001,100);
	//test_selfadjointeigensolver_rank(1,1001,100);
	
	return 0;
}