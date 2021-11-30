struct LJ{
private:
	double eps_,sigma_;
public:
	//==== constructors/destructors ====
	LJ(){}
	LJ(double eps, double sigma):eps_(eps),sigma_(sigma){}
	~LJ(){}
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& sigma(){return sigma_;}
	const double& sigma()const{return sigma_;}
	
	//==== operators ====
	double operator()(double r){
		const double x=sigma_/r;
		const double x6=x*x*x*x*x*x;
		return 4.0*eps_*(x6*x6-x6);
	}
};

void test_unit_struc();
void test_cell_list_square();