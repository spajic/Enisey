#ifndef __finder_pressure_pseudo_criticalI_h__
#define __finder_pressure_pseudo_criticalI_h__

#include <finder_pressure_pseudo_critical.h>

namespace Enisey
{

class FinderPressurePseudoCriticalI : virtual public FinderPressurePseudoCritical
{
public:

    virtual void Find(const ::Enisey::NumberSequence&,
                      const ::Enisey::NumberSequence&,
                      const ::Enisey::NumberSequence&,
                      ::Enisey::NumberSequence&,
                      const Ice::Current&);
};

}

#endif
