#pragma once

// forward-declarations
struct Passport;	
class Edge;		

class ManagerEdge
{
public:
  virtual void CountAll() = 0;
  virtual Edge* CreateEdge(const Passport* passport) = 0;
  virtual void FinishAddingEdges() = 0;
};