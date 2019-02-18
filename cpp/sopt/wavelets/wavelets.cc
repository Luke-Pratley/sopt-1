#include "sopt/wavelets/wavelets.h"
#ifdef SOPT_MPI
#include "sopt/wavelets/mpi_wavelets.h"
#endif
#include "sopt/config.h"
#include <exception>
#include "sopt/logging.h"

namespace sopt {
namespace wavelets {

Wavelet factory(std::string name, t_uint nlevels) {
  if (name == "dirac" or name == "Dirac") {
    SOPT_MEDIUM_LOG("Creating Dirac Wavelet");
    return Wavelet(daubechies_data(1), 0);
  }

  if (name.substr(0, 2) == "DB" or name.substr(0, 2) == "db") {
    std::istringstream sstr(name.substr(2, name.size() - 2));
    t_uint l(0);
    sstr >> l;
    SOPT_MEDIUM_LOG("Creating Daubechies Wavelet {}, level {}", l, nlevels);
    return Wavelet(daubechies_data(l), nlevels);
  }
  // Unknown input wavelet
  throw std::exception();
}

#ifdef SOPT_MPI
WaveletMPI factory(std::string name, t_uint nlevels, mpi::Communicator comm) {
  if (name == "dirac" or name == "Dirac") {
    SOPT_MEDIUM_LOG("Creating Dirac MPI Wavelet");
    return WaveletMPI(daubechies_data(1), 0, comm);
  }

  if (name.substr(0, 2) == "DB" or name.substr(0, 2) == "db") {
    std::istringstream sstr(name.substr(2, name.size() - 2));
    t_uint l(0);
    sstr >> l;
    SOPT_MEDIUM_LOG("Creating Daubechies MPI Wavelet {}, level {}", l, nlevels);
    return WaveletMPI(daubechies_data(l), nlevels, comm);
  }
  // Unknown input wavelet
  throw std::exception();
}
#endif
}  // namespace wavelets
}  // namespace sopt
