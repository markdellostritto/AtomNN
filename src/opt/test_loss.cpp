// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// eigen libraries
#include <Eigen/Dense>
// optimization
#include "src/opt/loss.hpp"
// string
#include "src/str/print.hpp"

void plot_loss(opt::Loss loss, const char* file){
	const double xmin=-6.0;
	const double xmax=6.0;
	const int N=300;
	const double dx=(xmax-xmin)/(N-1.0);
	FILE* writer=fopen(file,"w");
	Eigen::VectorXd value=Eigen::VectorXd::Constant(1,0.0);
	Eigen::VectorXd target=Eigen::VectorXd::Constant(1,0.0);
	Eigen::VectorXd grad=Eigen::VectorXd::Constant(1,0.0);
	if(writer!=NULL){
		for(int i=0; i<N; ++i){
			value[0]=xmin+i*dx;
			const double f=opt::Loss::error(loss,value,target,grad);
			fprintf(writer,"%f %f %f\n",value[0],f,grad[0]);
		}
		fclose(writer);
		writer=NULL;
	}
}

//**********************************************************************
// main
//**********************************************************************

int main(int argc, char* argv[]){
	
	//==== random ====
	std::srand(std::time(NULL));
	
	//==== optimizers ====
	const int DIM=2;
	char* str=new char[print::len_buf];
	
	//==== mae ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - PRINT - LOSS - MAE\n";
	plot_loss(opt::Loss::MAE,"loss_mae.dat");
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== mse ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - PRINT - LOSS - MSE\n";
	plot_loss(opt::Loss::MSE,"loss_mse.dat");
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== huber ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - PRINT - LOSS - HUBER\n";
	plot_loss(opt::Loss::HUBER,"loss_huber.dat");
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== asinh ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - PRINT - LOSS - ASINH\n";
	plot_loss(opt::Loss::ASINH,"loss_asinh.dat");
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== free memory ====
	delete[] str;
	
	return 0;
}
