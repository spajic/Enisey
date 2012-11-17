#ifndef _PARALLEL_MANAGER_ICE_I_ICE
#define _GAS_TRANSFER_SYSTEM_ICE_I_ICE

module Enisey {
	struct PassportPipeIce {
		double length;
		double dOuter;
		double dInner;
		double pMax;
		double pMin;
		double hydrEff;
		double roughCoeff;
		double heatExch;
		double tEnv;
	};
	struct WorkParamsIce {
		double denScIn;
		double co2In;
		double n2In;
		double pIn;
		double tIn;
		double pOut;
	};	 
	struct CalculatedParamsIce {
		double q;
		double dqDpIn;
		double dqDpOut;
		double tOut;
	};

	sequence<PassportPipeIce> PassportSequence;
	sequence<WorkParamsIce> WorkParamsSequence;
	sequence<CalculatedParamsIce> CalculatedParamsSequence;

	interface ParallelManagerIceI {
		void TakeUnderControl(PassportSequence passports);
		void SetWorkParams(WorkParamsSequence workParams);
		void CalculateAll();
		void GetCalculatedParams(out CalculatedParamsSequence calculatedParams);
	};
};

#endif