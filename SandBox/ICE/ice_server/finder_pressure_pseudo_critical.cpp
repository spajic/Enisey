// **********************************************************************
//
// Copyright (c) 2003-2011 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************
//
// Ice version 3.4.2
//
// <auto-generated>
//
// Generated from file `finder_pressure_pseudo_critical.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#include <finder_pressure_pseudo_critical.h>
#include <Ice/LocalException.h>
#include <Ice/ObjectFactory.h>
#include <Ice/BasicStream.h>
#include <IceUtil/Iterator.h>

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 304
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 > 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 2
#       error Ice patch level mismatch!
#   endif
#endif

static const ::std::string __Enisey__FinderPressurePseudoCritical__Find_name = "Find";

::Ice::Object* IceInternal::upCast(::Enisey::FinderPressurePseudoCritical* p) { return p; }
::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::Enisey::FinderPressurePseudoCritical* p) { return p; }

void
Enisey::__read(::IceInternal::BasicStream* __is, ::Enisey::FinderPressurePseudoCriticalPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::Enisey::FinderPressurePseudoCritical;
        v->__copyFrom(proxy);
    }
}

void
IceProxy::Enisey::FinderPressurePseudoCritical::Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, ::Enisey::NumberSequence& TemperaturePseudoCritical, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__Enisey__FinderPressurePseudoCritical__Find_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::Enisey::FinderPressurePseudoCritical* __del = dynamic_cast< ::IceDelegate::Enisey::FinderPressurePseudoCritical*>(__delBase.get());
            __del->Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, TemperaturePseudoCritical, __ctx);
            return;
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapperRelaxed(__delBase, __ex, true, __cnt);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, true, __cnt);
        }
    }
}

::Ice::AsyncResultPtr
IceProxy::Enisey::FinderPressurePseudoCritical::begin_Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, const ::Ice::Context* __ctx, const ::IceInternal::CallbackBasePtr& __del, const ::Ice::LocalObjectPtr& __cookie)
{
    __checkAsyncTwowayOnly(__Enisey__FinderPressurePseudoCritical__Find_name);
    ::IceInternal::OutgoingAsyncPtr __result = new ::IceInternal::OutgoingAsync(this, __Enisey__FinderPressurePseudoCritical__Find_name, __del, __cookie);
    try
    {
        __result->__prepare(__Enisey__FinderPressurePseudoCritical__Find_name, ::Ice::Idempotent, __ctx);
        ::IceInternal::BasicStream* __os = __result->__getOs();
        if(DensityInStandartConditions.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&DensityInStandartConditions[0], &DensityInStandartConditions[0] + DensityInStandartConditions.size());
        }
        if(Nitrogen.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&Nitrogen[0], &Nitrogen[0] + Nitrogen.size());
        }
        if(Hydrocarbon.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&Hydrocarbon[0], &Hydrocarbon[0] + Hydrocarbon.size());
        }
        __os->endWriteEncaps();
        __result->__send(true);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __result->__exceptionAsync(__ex);
    }
    return __result;
}

void
IceProxy::Enisey::FinderPressurePseudoCritical::end_Find(::Enisey::NumberSequence& TemperaturePseudoCritical, const ::Ice::AsyncResultPtr& __result)
{
    ::Ice::AsyncResult::__check(__result, this, __Enisey__FinderPressurePseudoCritical__Find_name);
    if(!__result->__wait())
    {
        try
        {
            __result->__throwUserException();
        }
        catch(const ::Ice::UserException& __ex)
        {
            throw ::Ice::UnknownUserException(__FILE__, __LINE__, __ex.ice_name());
        }
    }
    ::IceInternal::BasicStream* __is = __result->__getIs();
    __is->startReadEncaps();
    __is->read(TemperaturePseudoCritical);
    __is->endReadEncaps();
}

const ::std::string&
IceProxy::Enisey::FinderPressurePseudoCritical::ice_staticId()
{
    return ::Enisey::FinderPressurePseudoCritical::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::Enisey::FinderPressurePseudoCritical::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::Enisey::FinderPressurePseudoCritical);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::Enisey::FinderPressurePseudoCritical::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::Enisey::FinderPressurePseudoCritical);
}

::IceProxy::Ice::Object*
IceProxy::Enisey::FinderPressurePseudoCritical::__newInstance() const
{
    return new FinderPressurePseudoCritical;
}

void
IceDelegateM::Enisey::FinderPressurePseudoCritical::Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, ::Enisey::NumberSequence& TemperaturePseudoCritical, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __Enisey__FinderPressurePseudoCritical__Find_name, ::Ice::Idempotent, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        if(DensityInStandartConditions.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&DensityInStandartConditions[0], &DensityInStandartConditions[0] + DensityInStandartConditions.size());
        }
        if(Nitrogen.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&Nitrogen[0], &Nitrogen[0] + Nitrogen.size());
        }
        if(Hydrocarbon.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&Hydrocarbon[0], &Hydrocarbon[0] + Hydrocarbon.size());
        }
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    try
    {
        if(!__ok)
        {
            try
            {
                __og.throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                throw __uue;
            }
        }
        ::IceInternal::BasicStream* __is = __og.is();
        __is->startReadEncaps();
        __is->read(TemperaturePseudoCritical);
        __is->endReadEncaps();
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

void
IceDelegateD::Enisey::FinderPressurePseudoCritical::Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, ::Enisey::NumberSequence& TemperaturePseudoCritical, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, ::Enisey::NumberSequence& TemperaturePseudoCritical, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_DensityInStandartConditions(DensityInStandartConditions),
            _m_Nitrogen(Nitrogen),
            _m_Hydrocarbon(Hydrocarbon),
            _m_TemperaturePseudoCritical(TemperaturePseudoCritical)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::Enisey::FinderPressurePseudoCritical* servant = dynamic_cast< ::Enisey::FinderPressurePseudoCritical*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->Find(_m_DensityInStandartConditions, _m_Nitrogen, _m_Hydrocarbon, _m_TemperaturePseudoCritical, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        const ::Enisey::NumberSequence& _m_DensityInStandartConditions;
        const ::Enisey::NumberSequence& _m_Nitrogen;
        const ::Enisey::NumberSequence& _m_Hydrocarbon;
        ::Enisey::NumberSequence& _m_TemperaturePseudoCritical;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __Enisey__FinderPressurePseudoCritical__Find_name, ::Ice::Idempotent, __context);
    try
    {
        _DirectI __direct(DensityInStandartConditions, Nitrogen, Hydrocarbon, TemperaturePseudoCritical, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
}

::Ice::ObjectPtr
Enisey::FinderPressurePseudoCritical::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __Enisey__FinderPressurePseudoCritical_ids[2] =
{
    "::Enisey::FinderPressurePseudoCritical",
    "::Ice::Object"
};

bool
Enisey::FinderPressurePseudoCritical::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__Enisey__FinderPressurePseudoCritical_ids, __Enisey__FinderPressurePseudoCritical_ids + 2, _s);
}

::std::vector< ::std::string>
Enisey::FinderPressurePseudoCritical::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__Enisey__FinderPressurePseudoCritical_ids[0], &__Enisey__FinderPressurePseudoCritical_ids[2]);
}

const ::std::string&
Enisey::FinderPressurePseudoCritical::ice_id(const ::Ice::Current&) const
{
    return __Enisey__FinderPressurePseudoCritical_ids[0];
}

const ::std::string&
Enisey::FinderPressurePseudoCritical::ice_staticId()
{
    return __Enisey__FinderPressurePseudoCritical_ids[0];
}

::Ice::DispatchStatus
Enisey::FinderPressurePseudoCritical::___Find(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Idempotent, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Enisey::NumberSequence DensityInStandartConditions;
    ::Enisey::NumberSequence Nitrogen;
    ::Enisey::NumberSequence Hydrocarbon;
    __is->read(DensityInStandartConditions);
    __is->read(Nitrogen);
    __is->read(Hydrocarbon);
    __is->endReadEncaps();
    ::IceInternal::BasicStream* __os = __inS.os();
    ::Enisey::NumberSequence TemperaturePseudoCritical;
    Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, TemperaturePseudoCritical, __current);
    if(TemperaturePseudoCritical.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        __os->write(&TemperaturePseudoCritical[0], &TemperaturePseudoCritical[0] + TemperaturePseudoCritical.size());
    }
    return ::Ice::DispatchOK;
}

static ::std::string __Enisey__FinderPressurePseudoCritical_all[] =
{
    "Find",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

::Ice::DispatchStatus
Enisey::FinderPressurePseudoCritical::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__Enisey__FinderPressurePseudoCritical_all, __Enisey__FinderPressurePseudoCritical_all + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __Enisey__FinderPressurePseudoCritical_all)
    {
        case 0:
        {
            return ___Find(in, current);
        }
        case 1:
        {
            return ___ice_id(in, current);
        }
        case 2:
        {
            return ___ice_ids(in, current);
        }
        case 3:
        {
            return ___ice_isA(in, current);
        }
        case 4:
        {
            return ___ice_ping(in, current);
        }
    }

    assert(false);
    throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
}

void
Enisey::FinderPressurePseudoCritical::__write(::IceInternal::BasicStream* __os) const
{
    __os->writeTypeId(ice_staticId());
    __os->startWriteSlice();
    __os->endWriteSlice();
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
    Object::__write(__os);
#else
    ::Ice::Object::__write(__os);
#endif
}

void
Enisey::FinderPressurePseudoCritical::__read(::IceInternal::BasicStream* __is, bool __rid)
{
    if(__rid)
    {
        ::std::string myId;
        __is->readTypeId(myId);
    }
    __is->startReadSlice();
    __is->endReadSlice();
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
    Object::__read(__is, true);
#else
    ::Ice::Object::__read(__is, true);
#endif
}

// COMPILERFIX: Stream API is not supported with VC++ 6
#if !defined(_MSC_VER) || (_MSC_VER >= 1300)
void
Enisey::FinderPressurePseudoCritical::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type Enisey::FinderPressurePseudoCritical was not generated with stream support";
    throw ex;
}

void
Enisey::FinderPressurePseudoCritical::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type Enisey::FinderPressurePseudoCritical was not generated with stream support";
    throw ex;
}
#endif

void 
Enisey::__patch__FinderPressurePseudoCriticalPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::Enisey::FinderPressurePseudoCriticalPtr* p = static_cast< ::Enisey::FinderPressurePseudoCriticalPtr*>(__addr);
    assert(p);
    *p = ::Enisey::FinderPressurePseudoCriticalPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::Enisey::FinderPressurePseudoCritical::ice_staticId(), v->ice_id());
    }
}