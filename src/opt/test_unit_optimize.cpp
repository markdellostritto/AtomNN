// ann - optimization
#include "src/opt/test_unit_optimize.hpp"

int main(int argc, char* argv[]){
	
	//==== optimizers ====
	const int DIM=2;
	char* str=new char[print::len_buf];
	
	//==== sgd ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - SGD\n";
	Opt::SGD sgd(DIM);
	sgd.decay()=Opt::Decay::CONST;
	sgd.alpha()=1;
	sgd.gamma()=1e-3;
	opt_rosen(sgd);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== sdm ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - SDM\n";
	Opt::SDM sdm(DIM);
	sdm.decay()=Opt::Decay::CONST;
	sdm.alpha()=1;
	sdm.gamma()=1e-3;
	sdm.eta()=0.9;
	opt_rosen(sdm);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== nag ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - NAG\n";
	Opt::NAG nag(DIM);
	nag.decay()=Opt::Decay::CONST;
	nag.alpha()=1;
	nag.gamma()=1e-3;
	nag.eta()=0.9;
	opt_rosen(nag);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== adam ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - ADAM\n";
	Opt::ADAM adam(DIM);
	adam.decay()=Opt::Decay::CONST;
	adam.alpha()=1;
	adam.gamma()=1e-3;
	opt_rosen(adam);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== nadam ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - NADAM\n";
	Opt::NADAM nadam(DIM);
	nadam.decay()=Opt::Decay::CONST;
	nadam.alpha()=1;
	nadam.gamma()=1e-3;
	opt_rosen(nadam);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== amsgrad ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - AMSGRAD\n";
	Opt::AMSGRAD amsgrad(DIM);
	amsgrad.decay()=Opt::Decay::CONST;
	amsgrad.alpha()=1;
	amsgrad.gamma()=1e-3;
	opt_rosen(amsgrad);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== cg ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - CG\n";
	Opt::CG cg(DIM);
	cg.decay()=Opt::Decay::CONST;
	cg.alpha()=1;
	cg.gamma()=1e-3;
	opt_rosen(cg);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== free memory ====
	delete[] str;
	
	return 0;
}
