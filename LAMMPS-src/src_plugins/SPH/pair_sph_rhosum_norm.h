/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_PAIR_SPH_RhoSumNorm_H
#define LMP_PAIR_SPH_RhoSumNorm_H

#include "pair.h"
// #include <vector> // jsm

namespace LAMMPS_NS {

class PairSPHRhoSumNorm : public Pair {
 public:
  PairSPHRhoSumNorm(class LAMMPS *);
  ~PairSPHRhoSumNorm() override;
  void init_style() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  double single(int, int, int, int, double, double, double, double &) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 protected:
  double **cut;
  int nstep, first;
  int zmin;
  double rho0;
  int *fixrho_type;
  class Region *iregion; //jsm
  char *idregion; //jsm
  void allocate();
};

}    // namespace LAMMPS_NS

#endif