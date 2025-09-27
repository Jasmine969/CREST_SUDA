// clang-format off
/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 https://www.lammps.org/, Sandia National Laboratories
 LAMMPS development team: developers@lammps.org

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include "pair_sph_rho_mls.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include <vector>

using namespace LAMMPS_NS;

// declare LAPACK functions
extern "C" {
  void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
  void dpotri_(const char *uplo, const int *n, double *a, const int *lda, int *info);
}
/* ---------------------------------------------------------------------- */

PairSPHRhoMLS::PairSPHRhoMLS(LAMMPS *lmp) : Pair(lmp)
{
  if (atom->rho_flag != 1)
    error->all(FLERR, Error::NOLASTLINE,
               "Pair sph/rho/mls requires atom attribute density, e.g. in atom_style sph");

  restartinfo = 0;

  // set comm size needed by this Pair

  comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

PairSPHRhoMLS::~PairSPHRhoMLS()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(fixrho_type);
  }
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairSPHRhoMLS::init_style()
{
  // need a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

void PairSPHRhoMLS::compute(int eflag, int vflag)
{
  int i, j, ii, jj, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double rsq, imass, h, ih, ihsq;
  int *jlist;
  double wf;
  // neighbor list variables
  int inum, *ilist, *numneigh, **firstneigh;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double *rho = atom->rho;
  int *type = atom->type;
  double *mass = atom->mass;
  int dim = domain->dimension;
  const int Adim = dim + 1;
  const int matSize = Adim * Adim;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  std::vector<double> numrt(inum), denom(inum); // jsm
  int coord; // jsm
  bool mls_flag;
  std::vector<bool> modify_rho_flag(inum, true); // jsm
  double matA[matSize]; // jsm
  double volj, jmass, wVolj, p_base[4]; // jsm
  int a, b, lapack_error;
  // recompute density
  // we use a full neighborlist here

  if (nstep != 0) {
    if ((update->ntimestep % nstep) == 0) {

      // initialize density with self-contribution,
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        itype = type[i];
        if (fixrho_type[itype]) continue;
        imass = mass[itype];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        jlist = firstneigh[i];
        jnum = numneigh[i];
        coord = 0;
        modify_rho_flag[ii] = true;
        // calculate coordination
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;

          jtype = type[j];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];

          rsq = delx * delx + dely * dely + delz * delz;

          if (rsq < cutsq[itype][jtype]) ++coord;
        }
        if (coord < zmin) {
            modify_rho_flag[ii] = false;
            continue;
        }
        // coord >= zmin ===================================
        // MLS of particle i
        // establish the moment matrix
        // Zero upper-triangle M and cut (will be symmetric):
        for (a = 0; a < Adim; a++) {
          for (b = a; b < Adim; b++) {
            matA[a * Adim + b] = 0;
          }
        }
        // self-contribution
        h = cut[itype][itype];
        if (dim == 3) {
          wf = 2.1541870227086614782 / (h * h * h);  // 3D
        } else {
          wf = 1.5915494309189533576e0 / (h * h); // 2D
        }
        matA[0] += wf * imass / rho[i];
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;

          jtype = type[j];
          jmass = mass[jtype];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];

          rsq = delx * delx + dely * dely + delz * delz;

          if (rsq < cutsq[itype][jtype]) {
            h = cut[itype][jtype];
            ih = 1.0 / h;
            ihsq = ih * ih;
            if (dim == 3) {
              // 3D
              wf = 1.0 - rsq * ihsq;
              wf = wf * wf;
              wf = wf * wf;
              wf = 2.1541870227086614782e0 * wf * ihsq * ih;
            } else {
              // 2D
              wf = 1.0 - rsq * ihsq;
              wf = wf * wf;
              wf = wf * wf;
              wf = 1.5915494309189533576e0 * wf * ihsq;
            }
            wVolj = wf * jmass / rho[j];
            p_base[0] = 1;
            p_base[1] = delx;
            p_base[2] = dely;
            p_base[3] = delz;
            for (a = 0; a < Adim; a++) {
              for (b = a; b < Adim; b++) {
                matA[a * Adim + b] += p_base[a] * p_base[b] * wVolj;
              }
            }
          }
        }
        // Populate the lower triangle from the symmetric entries of M:
        for (a = 0; a < Adim; a++) {
          for (b = a; b < Adim; b++) {
            matA[b * Adim + a] = matA[a * Adim + b];
          }
        }
        // calculate the inverse
        // Use LAPACK to get Ainv, use Cholesky decomposition since the
        // polynomials are independent, A is symmetrix & positive-definite
        const char uplo = 'U';
        dpotrf_(&uplo, &Adim, matA, &Adim, &lapack_error);
        if (lapack_error) {
          // check if not positive-definite
          if (lapack_error > 0)
          error->warning(FLERR, "Failed DPOTRF2 decomposition in sph/rho/mls, info = {}",
                         lapack_error);
          // Revert to uncorrected SPH for this particle
          mls_flag = false;
        } else {
          mls_flag = true;
          // matA is now its inverse
          dpotri_(&uplo, &Adim, matA, &Adim, &lapack_error);
        }

        /* ---------------------------------------------------------
        compute numerator and denominator 
        ----------------------------------------------------------*/
        // calculate self contribution
        h = cut[itype][itype];
        if (dim == 3) {
          wf = 2.1541870227086614782 / (h * h * h);  // 3D
        } else {
          wf = 1.5915494309189533576e0 / (h * h); // 2D
        }
        if (mls_flag) wf *= matA[0];
        // jsm
        numrt[ii] = imass * wf;
        denom[ii] = imass / rho[i] * wf;

        // add density at each atom via kernel function overlap
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;

          jtype = type[j];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx * delx + dely * dely + delz * delz;

          if (rsq < cutsq[itype][jtype]) {
            h = cut[itype][jtype];
            ih = 1.0 / h;
            ihsq = ih * ih;

            if (dim == 3) {
              // 3D
              wf = 1.0 - rsq * ihsq;
              wf = wf * wf;
              wf = wf * wf;
              wf = 2.1541870227086614782e0 * wf * ihsq * ih;
            } else {
              // 2D
              wf = 1.0 - rsq * ihsq;
              wf = wf * wf;
              wf = wf * wf;
              wf = 1.5915494309189533576e0 * wf * ihsq;
            }
            if (mls_flag) {
              if (dim == 3) 
                wf *= matA[0] + matA[Adim] * delx + matA[Adim*2] * dely + matA[Adim*3] * delz;
              else 
                wf *= matA[0] + matA[Adim] * delx + matA[Adim*2] * dely;
            }
            numrt[ii] += jmass * wf;
            denom[ii] += jmass / rho[j] * wf;
          }
        }
      }
      // update density
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        itype = type[i];
        if (fixrho_type[itype]) continue;
        if (modify_rho_flag[ii])
          rho[i] = numrt[ii]/denom[ii];
        else
          rho[i] = rho0;
      }
    }
  }

  // communicate densities
  comm->forward_comm(this);
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSPHRhoMLS::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(fixrho_type, n + 1, "rheo:fixrho_type");
  for (int i = 1; i <= n; i++) fixrho_type[i] = 0; // default: rho of all types should be updated 
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairSPHRhoMLS::settings(int narg, char **arg)
{
  if (narg != 1)
    error->all(FLERR, Error::NOLASTLINE,
        "Illegal number of arguments for pair_style sph/rho/mls");
  nstep = utils::inumeric(FLERR,arg[0],false,lmp);
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHRhoMLS::coeff(int narg, char **arg)
{
  if (narg < 3)
    error->all(FLERR,"Incorrect number of args for sph/rho/mls coefficients" + utils::errorurl(21));
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR,arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR,arg[1], 1, atom->ntypes, jlo, jhi, error);

  double cut_one = utils::numeric(FLERR,arg[2],false,lmp);
  zmin = 0;
  int iarg = 3;
  double eps_one = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "zmin/rho0") == 0) {
      if (iarg + 3 > narg) utils::missing_cmd_args(FLERR, "pair sph/rho/mls", error);
      zmin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      rho0 = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "fixed/rho") == 0) {
      int n = atom->ntypes;
      if (iarg + n >= narg) utils::missing_cmd_args(FLERR, "pair sph/rho/mls fixed/rho", error);
      for (int i = 1; i <= n; i++) fixrho_type[i] = utils::numeric(FLERR, arg[iarg + i], false, lmp);
      iarg += n;
    }
    else
      error->all(FLERR, "Unknown pair sph/rho/mls keyword: {}", arg[iarg]);
    iarg += 1;
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHRhoMLS::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair sph/rho/mls coeffs are not set. Status:\n"
               + Info::get_pair_coeff_status(lmp));

  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHRhoMLS::single(int /*i*/, int /*j*/, int /*itype*/, int /*jtype*/, double /*rsq*/,
    double /*factor_coul*/, double /*factor_lj*/, double &fforce)
{
  fforce = 0.0;

  return 0.0;
}

/* ---------------------------------------------------------------------- */

int PairSPHRhoMLS::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;
  double *rho = atom->rho;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = rho[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairSPHRhoMLS::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;
  double *rho = atom->rho;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
    rho[i] = buf[m++];
}
