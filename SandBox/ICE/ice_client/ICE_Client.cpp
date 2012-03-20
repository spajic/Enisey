// ICE_Client.cpp : Defines the entry point for the console application.
#include <finder_temperature_pseudo_critical.h>
#include "Ice/Ice.h"

int main(int argc, char* argv[]) {
	int status = 0;
	Ice::CommunicatorPtr ic;
	try
	{
		//
		// Create a communicator
		//
		ic = Ice::initialize(argc, argv);

		//
		// Create a proxy for the root directory
		//
		Ice::ObjectPrx base = ic->stringToProxy("alex:default -p 10000");

		Enisey::FinderTemperaturePseudoCriticalPrx finder_proxy = 
			Enisey::FinderTemperaturePseudoCriticalPrx::checkedCast(base);
		if(!finder_proxy) {
			throw "Invalid proxy";
		}

		Enisey::NumberSequence Densities(3, 1);
		Enisey::NumberSequence Nitrogens(3, 1);
		Enisey::NumberSequence Hydrocarbones(3, 1);
		Enisey::NumberSequence Temperatures(3, 1);
		finder_proxy->Find(Densities, Nitrogens, Hydrocarbones, Temperatures);
		for(unsigned int i = 0; i < Temperatures.size(); ++i) {
			std::cout << "Temps[" << i << "] = " << Temperatures[i] << std::endl;
		}
	}
	catch(const Ice::Exception& ex)
	{
		std::cerr << ex << std::endl;
		status = 1;
	}
	catch(const char* msg)
	{
		std::cerr << msg << std::endl;
		status = 1;
	}

	//
	// Clean up
	//
	if(ic)
	{
		try
		{
			ic->destroy();
		}
		catch(const Ice::Exception& e)
		{
			std::cerr << e << std::endl;
			status = 1;
		}
	}

	return status;
}

