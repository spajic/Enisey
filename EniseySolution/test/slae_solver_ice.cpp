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
// Generated from file `slae_solver_ice.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#include <slae_solver_ice.h>
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

static const ::std::string __Enisey__SlaeSolverIce__Solve_name = "Solve";

::Ice::Object* IceInternal::upCast(::Enisey::SlaeSolverIce* p) { return p; }
::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::Enisey::SlaeSolverIce* p) { return p; }

void
Enisey::__read(::IceInternal::BasicStream* __is, ::Enisey::SlaeSolverIcePrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::Enisey::SlaeSolverIce;
        v->__copyFrom(proxy);
    }
}

void
IceProxy::Enisey::SlaeSolverIce::Solve(const ::Enisey::IntSequence& AIndexes, const ::Enisey::DoubleSequence& AValues, const ::Enisey::DoubleSequence& B, ::Enisey::DoubleSequence& X, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__Enisey__SlaeSolverIce__Solve_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::Enisey::SlaeSolverIce* __del = dynamic_cast< ::IceDelegate::Enisey::SlaeSolverIce*>(__delBase.get());
            __del->Solve(AIndexes, AValues, B, X, __ctx);
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
IceProxy::Enisey::SlaeSolverIce::begin_Solve(const ::Enisey::IntSequence& AIndexes, const ::Enisey::DoubleSequence& AValues, const ::Enisey::DoubleSequence& B, const ::Ice::Context* __ctx, const ::IceInternal::CallbackBasePtr& __del, const ::Ice::LocalObjectPtr& __cookie)
{
    __checkAsyncTwowayOnly(__Enisey__SlaeSolverIce__Solve_name);
    ::IceInternal::OutgoingAsyncPtr __result = new ::IceInternal::OutgoingAsync(this, __Enisey__SlaeSolverIce__Solve_name, __del, __cookie);
    try
    {
        __result->__prepare(__Enisey__SlaeSolverIce__Solve_name, ::Ice::Idempotent, __ctx);
        ::IceInternal::BasicStream* __os = __result->__getOs();
        if(AIndexes.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&AIndexes[0], &AIndexes[0] + AIndexes.size());
        }
        if(AValues.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&AValues[0], &AValues[0] + AValues.size());
        }
        if(B.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&B[0], &B[0] + B.size());
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
IceProxy::Enisey::SlaeSolverIce::end_Solve(::Enisey::DoubleSequence& X, const ::Ice::AsyncResultPtr& __result)
{
    ::Ice::AsyncResult::__check(__result, this, __Enisey__SlaeSolverIce__Solve_name);
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
    __is->read(X);
    __is->endReadEncaps();
}

const ::std::string&
IceProxy::Enisey::SlaeSolverIce::ice_staticId()
{
    return ::Enisey::SlaeSolverIce::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::Enisey::SlaeSolverIce::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::Enisey::SlaeSolverIce);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::Enisey::SlaeSolverIce::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::Enisey::SlaeSolverIce);
}

::IceProxy::Ice::Object*
IceProxy::Enisey::SlaeSolverIce::__newInstance() const
{
    return new SlaeSolverIce;
}

void
IceDelegateM::Enisey::SlaeSolverIce::Solve(const ::Enisey::IntSequence& AIndexes, const ::Enisey::DoubleSequence& AValues, const ::Enisey::DoubleSequence& B, ::Enisey::DoubleSequence& X, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __Enisey__SlaeSolverIce__Solve_name, ::Ice::Idempotent, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        if(AIndexes.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&AIndexes[0], &AIndexes[0] + AIndexes.size());
        }
        if(AValues.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&AValues[0], &AValues[0] + AValues.size());
        }
        if(B.size() == 0)
        {
            __os->writeSize(0);
        }
        else
        {
            __os->write(&B[0], &B[0] + B.size());
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
        __is->read(X);
        __is->endReadEncaps();
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

void
IceDelegateD::Enisey::SlaeSolverIce::Solve(const ::Enisey::IntSequence& AIndexes, const ::Enisey::DoubleSequence& AValues, const ::Enisey::DoubleSequence& B, ::Enisey::DoubleSequence& X, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(const ::Enisey::IntSequence& AIndexes, const ::Enisey::DoubleSequence& AValues, const ::Enisey::DoubleSequence& B, ::Enisey::DoubleSequence& X, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_AIndexes(AIndexes),
            _m_AValues(AValues),
            _m_B(B),
            _m_X(X)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::Enisey::SlaeSolverIce* servant = dynamic_cast< ::Enisey::SlaeSolverIce*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->Solve(_m_AIndexes, _m_AValues, _m_B, _m_X, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        const ::Enisey::IntSequence& _m_AIndexes;
        const ::Enisey::DoubleSequence& _m_AValues;
        const ::Enisey::DoubleSequence& _m_B;
        ::Enisey::DoubleSequence& _m_X;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __Enisey__SlaeSolverIce__Solve_name, ::Ice::Idempotent, __context);
    try
    {
        _DirectI __direct(AIndexes, AValues, B, X, __current);
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
Enisey::SlaeSolverIce::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __Enisey__SlaeSolverIce_ids[2] =
{
    "::Enisey::SlaeSolverIce",
    "::Ice::Object"
};

bool
Enisey::SlaeSolverIce::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__Enisey__SlaeSolverIce_ids, __Enisey__SlaeSolverIce_ids + 2, _s);
}

::std::vector< ::std::string>
Enisey::SlaeSolverIce::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__Enisey__SlaeSolverIce_ids[0], &__Enisey__SlaeSolverIce_ids[2]);
}

const ::std::string&
Enisey::SlaeSolverIce::ice_id(const ::Ice::Current&) const
{
    return __Enisey__SlaeSolverIce_ids[0];
}

const ::std::string&
Enisey::SlaeSolverIce::ice_staticId()
{
    return __Enisey__SlaeSolverIce_ids[0];
}

::Ice::DispatchStatus
Enisey::SlaeSolverIce::___Solve(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Idempotent, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Enisey::IntSequence AIndexes;
    ::Enisey::DoubleSequence AValues;
    ::Enisey::DoubleSequence B;
    __is->read(AIndexes);
    __is->read(AValues);
    __is->read(B);
    __is->endReadEncaps();
    ::IceInternal::BasicStream* __os = __inS.os();
    ::Enisey::DoubleSequence X;
    Solve(AIndexes, AValues, B, X, __current);
    if(X.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        __os->write(&X[0], &X[0] + X.size());
    }
    return ::Ice::DispatchOK;
}

static ::std::string __Enisey__SlaeSolverIce_all[] =
{
    "Solve",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

::Ice::DispatchStatus
Enisey::SlaeSolverIce::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__Enisey__SlaeSolverIce_all, __Enisey__SlaeSolverIce_all + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __Enisey__SlaeSolverIce_all)
    {
        case 0:
        {
            return ___Solve(in, current);
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
Enisey::SlaeSolverIce::__write(::IceInternal::BasicStream* __os) const
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
Enisey::SlaeSolverIce::__read(::IceInternal::BasicStream* __is, bool __rid)
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
Enisey::SlaeSolverIce::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type Enisey::SlaeSolverIce was not generated with stream support";
    throw ex;
}

void
Enisey::SlaeSolverIce::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type Enisey::SlaeSolverIce was not generated with stream support";
    throw ex;
}
#endif

void 
Enisey::__patch__SlaeSolverIcePtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::Enisey::SlaeSolverIcePtr* p = static_cast< ::Enisey::SlaeSolverIcePtr*>(__addr);
    assert(p);
    *p = ::Enisey::SlaeSolverIcePtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::Enisey::SlaeSolverIce::ice_staticId(), v->ice_id());
    }
}
