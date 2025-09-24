#ifdef PAIR_CLASS
// clang-format off
PairStyle(sph/rho/mls,PairSPHRhoMLS);
// clang-format on
#else

#ifndef LMP_PAIR_SPH_RHO_MLS_H
#define LMP_PAIR_SPH_RHO_MLS_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSPHRhoMLS : public Pair {
 public:
  PairSPHRhoMLS(class LAMMPS *);
  ~PairSPHRhoMLS();
  void init_style() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  double single(int, int, int, int, double, double, double, double &) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  
 protected:
  // MLS parameters
  double **cut;
  int nstep;
  int zmin; // jsm
  double rho0; // jsm
  int *fixrho_type;
  
  void allocate();
};

} // namespace LAMMPS_NS

#endif

#endif