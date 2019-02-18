#ifndef SOPT_WAVELETS_DIRECT_H
#define SOPT_WAVELETS_DIRECT_H

#include "sopt/config.h"
#include <type_traits>
#include "sopt/types.h"
#include "sopt/wavelets/wavelet_data.h"
#ifdef SOPT_MPI
#include "sopt/mpi/communicator.h"
#endif

// Function inside anonymouns namespace won't appear in library
namespace sopt {
namespace wavelets {

namespace {
//! \brief Single-level 1d direct transform
//! \param[out] coeffs_: output of the function (despite the const)
//! \param[in] signal: input signal for which to compute wavelet transform
//! \param[in] wavelet: contains wavelet coefficients
template <class T0, class T1>
typename std::enable_if<T1::IsVectorAtCompileTime, void>::type direct_transform_impl(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal,
    WaveletData const &wavelet) {
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);
  assert(coeffs.size() == signal.size());
  assert(wavelet.direct_filter.low.size() == wavelet.direct_filter.high.size());

  auto const N = signal.size() / 2;
  down_convolve(coeffs.head(N), signal, wavelet.direct_filter.low);
  down_convolve(coeffs.tail(coeffs.size() - N), signal, wavelet.direct_filter.high);
}

//! Single-level 2d direct transform
//! \param[out] coeffs_: output of the function (despite the const)
//! \param[inout] signal: input signal for which to compute wavelet transform. Input is modified.
//! \param[in] wavelet: contains wavelet coefficients
template <class T0, class T1>
typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type direct_transform_impl(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal_,
    WaveletData const &wavelet) {
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);
  Eigen::ArrayBase<T1> &signal = const_cast<Eigen::ArrayBase<T1> &>(signal_);
  assert(coeffs.rows() == signal.rows());
  assert(coeffs.cols() == signal.cols());
  assert(wavelet.direct_filter.low.size() == wavelet.direct_filter.high.size());
  for (t_uint i = 0; i < static_cast<t_uint>(coeffs.rows()); ++i)
    direct_transform_impl(coeffs.row(i).transpose(), signal.row(i).transpose(), wavelet);

  for (t_uint i = 0; i < static_cast<t_uint>(coeffs.cols()); ++i) {
    signal.col(i) = coeffs.col(i);
    direct_transform_impl(coeffs.col(i), signal.col(i), wavelet);
  }
}
#ifdef SOPT_MPI
//! Single-level 2d direct transform with MPI
//! \param[out] coeffs_: output of the function (despite the const)
//! \param[inout] signal: input signal for which to compute wavelet transform. Input is modified.
//! \param[in] wavelet: contains wavelet coefficients
//! \param[in] comm: MPI communicator
template <class T0, class T1>
typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type direct_transform_impl(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal_,
    WaveletData const &wavelet, const mpi::Communicator &comm) {
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);
  Eigen::ArrayBase<T1> &signal = const_cast<Eigen::ArrayBase<T1> &>(signal_);
  assert(coeffs.rows() == signal.rows());
  assert(coeffs.cols() == signal.cols());
  assert(wavelet.direct_filter.low.size() == wavelet.direct_filter.high.size());
  t_uint loc = 0;

  const t_uint row_r = coeffs.rows() % comm.size();
  const t_uint row_partition = static_cast<t_uint>(std::floor(coeffs.rows() / comm.size())) +
                               ((row_r > comm.rank()) ? 1 : 0);
  const t_uint start_row =
      std::max<t_int>({static_cast<t_int>(comm.rank() - row_r), 0}) *
          static_cast<t_uint>(std::floor(coeffs.rows() / comm.size())) +
      std::min<t_int>({static_cast<t_int>(row_r), static_cast<t_int>(comm.rank())}) *
          (static_cast<t_uint>(std::floor(coeffs.rows() / comm.size())) + 1);

  for (t_uint i = start_row; i < row_partition + start_row; ++i)
    direct_transform_impl(coeffs.row(i).transpose(), signal.row(i).transpose(), wavelet);
  loc = 0;
  for (t_uint i = 0; i < comm.size(); ++i) {
    const t_uint part = comm.broadcast(row_partition, i);
    coeffs.block(loc, 0, part, coeffs.cols()) =
        comm.broadcast<Image<typename T0::Scalar>>(coeffs.block(loc, 0, part, coeffs.cols()), i);
    loc += part;
  }
  const t_uint col_r = coeffs.cols() % comm.size();
  const t_uint col_partition = static_cast<t_uint>(std::floor(coeffs.cols() / comm.size())) +
                               ((col_r > comm.rank()) ? 1 : 0);
  const t_uint start_col =
      std::max<t_int>({static_cast<t_int>(comm.rank() - col_r), 0}) *
          static_cast<t_uint>(std::floor(coeffs.cols() / comm.size())) +
      std::min<t_int>({static_cast<t_int>(col_r), static_cast<t_int>(comm.rank())}) *
          (static_cast<t_uint>(std::floor(coeffs.cols() / comm.size())) + 1);

  for (t_uint i = start_col; i < col_partition + start_col; ++i) {
    signal.col(i) = coeffs.col(i);
    direct_transform_impl(coeffs.col(i), signal.col(i), wavelet);
  }
  loc = 0;
  for (t_uint i = 0; i < comm.size(); ++i) {
    t_uint const part = comm.broadcast(col_partition, i);
    coeffs.block(0, loc, coeffs.rows(), part) =
        comm.broadcast<Image<typename T0::Scalar>>(coeffs.block(0, loc, coeffs.rows(), part), i);
    loc += part;
  }
}
#endif

}  // namespace

//! \brief N-levels 1d direct transform
//! \param[out] coeffs_: output of the function (despite the const)
//! \param[in] signal: input signal for which to compute wavelet transfor
//! \param[in] wavelet: contains wavelet coefficients
//! \note The size  of the coefficients should a multiple of $2^l$ where $l$ is the number of
//! levels.
template <class T0, class T1>
typename std::enable_if<T1::IsVectorAtCompileTime, void>::type direct_transform(
    Eigen::ArrayBase<T0> &coeffs, Eigen::ArrayBase<T1> const &signal, t_uint levels,
    WaveletData const &wavelet) {
  assert(coeffs.rows() == signal.rows());
  assert(coeffs.cols() == signal.cols());

  auto input = copy(signal);
  if (levels > 0) direct_transform_impl(coeffs, input, wavelet);
  for (t_uint level(1); level < levels; ++level) {
    auto const N = static_cast<t_uint>(signal.size()) >> level;
    input.head(N) = coeffs.head(N);
    direct_transform_impl(coeffs.head(N), input.head(N), wavelet);
  }
}
//! N-levels 2d direct transform
//! \param[in] signal: input signal for which to compute wavelet transfor
//! \param[in] wavelet: contains wavelet coefficients
//! \note The size  of the coefficients should a multiple of $2^l$ where $l$ is the number of
//! levels.
template <class T0, class T1>
typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type direct_transform(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal, t_uint levels,
    WaveletData const &wavelet) {
  assert(coeffs_.rows() == signal.rows());
  assert(coeffs_.cols() == signal.cols());
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);

  if (levels == 0) {
    coeffs = signal;
    return;
  }

  auto input = copy(signal);
  direct_transform_impl(coeffs, input, wavelet);
  for (t_uint level(1); level < levels; ++level) {
    auto const Nx = static_cast<t_uint>(signal.rows()) >> level;
    auto const Ny = static_cast<t_uint>(signal.cols()) >> level;
    input.topLeftCorner(Nx, Ny) = coeffs.topLeftCorner(Nx, Ny);
    direct_transform_impl(coeffs.topLeftCorner(Nx, Ny), input.topLeftCorner(Nx, Ny), wavelet);
  }
}
#ifdef SOPT_MPI
//! N-levels 2d direct transform with MPI
//! \param[in] signal: input signal for which to compute wavelet transfor
//! \param[in] wavelet: contains wavelet coefficients
//! \note The size  of the coefficients should a multiple of $2^l$ where $l$ is the number of
//! levels.
template <class T0, class T1>
typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type direct_transform(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal, t_uint levels,
    WaveletData const &wavelet, const mpi::Communicator &comm) {
  assert(coeffs_.rows() == signal.rows());
  assert(coeffs_.cols() == signal.cols());
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);

  if (levels == 0) {
    coeffs = signal;
    return;
  }

  auto input = copy(signal);
  direct_transform_impl(coeffs, input, wavelet, comm);
  for (t_uint level(1); level < levels; ++level) {
    auto const Nx = static_cast<t_uint>(signal.rows()) >> level;
    auto const Ny = static_cast<t_uint>(signal.cols()) >> level;
    input.topLeftCorner(Nx, Ny) = coeffs.topLeftCorner(Nx, Ny);
    direct_transform_impl(coeffs.topLeftCorner(Nx, Ny), input.topLeftCorner(Nx, Ny), wavelet, comm);
  }
}
//! \brief Direct 1d and 2d transform with MPI
//! \note The size  of the coefficients should a multiple of $2^l$ where $l$ is the number of
//! levels.
template <class T0>
auto direct_transform(Eigen::ArrayBase<T0> const &signal, t_uint levels, WaveletData const &wavelet,
                      const mpi::Communicator &comm) -> decltype(copy(signal)) {
  auto result = copy(signal);
  direct_transform(result, signal, levels, wavelet, comm);
  return result;
}
#endif

//! \brief Direct 1d and 2d transform
//! \note The size  of the coefficients should a multiple of $2^l$ where $l$ is the number of
//! levels.
template <class T0>
auto direct_transform(Eigen::ArrayBase<T0> const &signal, t_uint levels, WaveletData const &wavelet)
    -> decltype(copy(signal)) {
  auto result = copy(signal);
  direct_transform(result, signal, levels, wavelet);
  return result;
}
}  // namespace wavelets
}  // namespace sopt
#endif
