#include "Variational.h"

void VariationalRefine(const cv::Mat& fixed_image, const cv::Mat& moved_image, cv::Mat& flow_image)
{
	cv::Ptr<cv::VariationalRefinement> vf = cv::VariationalRefinement::create();

	//vf->setAlpha(1.0f);					// Weight of the smoothness term
	//vf->setGamma(0.71f);				// Weight of the gradient constancy term
	//vf->setDelta(0.0f);					// Weight of the color constancy term
	//vf->setOmega(1.9f);					// Relaxation factor in SOR
	//vf->setFixedPointIterations(5);		// Number of outer (fixed-point) iterations in the minimization procedure
	///*Number of inner successive over - relaxation(SOR) iterations
	//in the minimization procedure to solve the respective linear system*/
	//vf->setSorIterations(30);

	// 	vf->setAlpha(20);	// smooth term
	// 	vf->setGamma(10);	// gradient term
	// 	vf->setDelta(5);	// color term
	//	vf->setOmega(1.9);	// Relaxation factor in SOR
	// 	vf->setFixedPointIterations(5);
	// 	vf->setSorIterations(5);

	std::cout << "	Alpha: " << vf->getAlpha() << std::endl;
	std::cout << "	Gamma: " << vf->getGamma() << std::endl;
	std::cout << "	Delta: " << vf->getDelta() << std::endl;
	std::cout << "	Omega: " << vf->getOmega() << std::endl;
	std::cout << "	FixedPointIterations: " << vf->getFixedPointIterations() << std::endl;
	std::cout << "	SorIterations: " << vf->getSorIterations() << std::endl;

	vf->calc(fixed_image, moved_image, flow_image);
}
