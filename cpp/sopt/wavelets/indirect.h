#ifndef SOPT_WAVELET_INDIRECT_H
#define SOPT_WAVELET_INDIRECT_H

#include "sopt/config.h"
#include "sopt/types.h"
#include "sopt/wavelets/innards.impl.h"
#include "sopt/wavelets/wavelet_data.h"

// Function inside anonymouns namespace won't appear in library
namespace sopt {
namespace wavelets {
namespace {
//! Single-level 1d indirect transform
//! \param[in] coeffs_: input coefficients
//! \param[out] signal: output with the reconstituted signal
//! \param[in] wavelet: contains wavelet coefficients
template <class T0, class T1>
typename std::enable_if<T1::IsVectorAtCompileTime, void>::type indirect_transform_impl(
    Eigen::ArrayBase<T0> const &coeffs, Eigen::ArrayBase<T1> const &signal_,
    WaveletData const &wavelet) {
  Eigen::ArrayBase<T1> &signal = const_cast<Eigen::ArrayBase<T1> &>(signal_);

  assert(coeffs.size() == signal.size());
  assert(coeffs.size() % 2 == 0);

  up_convolve_sum(signal, coeffs, wavelet.indirect_filter.low_even, wavelet.indirect_filter.low_odd,
                  wavelet.indirect_filter.high_even, wavelet.indirect_filter.high_odd);
}
//! Single-level 2d indirect transform
//! \param[in] coeffs_: input coefficients
//! \param[out] signal: output with the reconstituted signal
//! \param[in] wavelet: contains wavelet coefficients
template <class T0, class T1>
typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type indirect_transform_impl(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal_,
    WaveletData const &wavelet) {
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);
  Eigen::ArrayBase<T1> &signal = const_cast<Eigen::ArrayBase<T1> &>(signal_);
  assert(coeffs.rows() == signal.rows() and coeffs.cols() == signal.cols());
  assert(coeffs.rows() % 2 == 0 and coeffs.cols() % 2 == 0);

  for (t_uint i = 0; i < signal.rows(); ++i)
    indirect_transform_impl(coeffs.row(i).transpose(), signal.row(i).transpose(), wavelet);
  coeffs = signal;
  for (t_uint j = 0; j < signal.cols(); ++j)
    indirect_transform_impl(coeffs.col(j), signal.col(j), wavelet);
}
#ifdef SOPT_MPI
//! Single-level 2d direct transform with MPI
//! \param[out] coeffs_: output of the function (despite the const)
//! \param[inout] signal: input signal for which to compute wavelet transform. Input is modified.
//! \param[in] wavelet: contains wavelet coefficients
//! \param[in] comm: MPI communicator
template <class T0, class T1>
typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type indirect_transform_impl(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal_,
    WaveletData const &wavelet, const mpi::Communicator &comm) {
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);
  Eigen::ArrayBase<T1> &signal = const_cast<Eigen::ArrayBase<T1> &>(signal_);
  assert(coeffs.rows() == signal.rows() and coeffs.cols() == signal.cols());
  assert(coeffs.rows() % 2 == 0 and coeffs.cols() % 2 == 0);
  t_uint loc = 0;

  const t_uint row_r = signal.rows() % comm.size();
  const t_uint row_partition = static_cast<t_uint>(std::floor(signal.rows() / comm.size())) +
                               ((row_r > comm.rank()) ? 1 : 0);
  const t_uint start_row =
      std::max<t_int>({static_cast<t_int>(comm.rank() - row_r), 0}) *
          static_cast<t_uint>(std::floor(signal.rows() / comm.size())) +
      std::min<t_int>({static_cast<t_int>(row_r), static_cast<t_int>(comm.rank())}) *
          (static_cast<t_uint>(std::floor(signal.rows() / comm.size())) + 1);

  for (t_uint i = start_row; i < row_partition + start_row; ++i)
    indirect_transform_impl(coeffs.row(i).transpose(), signal.row(i).transpose(), wavelet);
  loc = 0;
  for (t_uint i = 0; i < comm.size(); ++i) {
    const t_uint part = comm.broadcast(row_partition, i);
    signal.block(loc, 0, part, signal.cols()) =
        comm.broadcast<T0>(signal.block(loc, 0, part, signal.cols()), i);
    loc += part;
  }
  coeffs = signal;
  const t_uint col_r = signal.cols() % comm.size();
  const t_uint col_partition = static_cast<t_uint>(std::floor(signal.cols() / comm.size())) +
                               ((col_r > comm.rank()) ? 1 : 0);
  const t_uint start_col =
      std::max<t_int>({static_cast<t_int>(comm.rank() - col_r), 0}) *
          static_cast<t_uint>(std::floor(signal.cols() / comm.size())) +
      std::min<t_int>({static_cast<t_int>(col_r), static_cast<t_int>(comm.rank())}) *
          (static_cast<t_uint>(std::floor(signal.cols() / comm.size())) + 1);
  for (t_uint i = start_col; i < col_partition + start_col; ++i)
    indirect_transform_impl(coeffs.col(i), signal.col(i), wavelet);
  loc = 0;
  for (t_uint i = 0; i < comm.size(); ++i) {
    t_uint const part = comm.broadcast(col_partition, i);
    signal.block(0, loc, signal.rows(), part) =
        comm.broadcast<T0>(signal.block(0, loc, signal.rows(), part), i);
    loc += part;
  }
}
#endif
}  // namespace

//! \brief N-levels 1d indirect transform
//! \param[in] coeffs_: input coefficients
//! \param[out] signal: output with the reconstituted signal
//! \param[in] wavelet: contains wavelet coefficients
//! \note The size  of the coefficients should a multiple of $2^l$ where $l$ is the number of
//! levels.
template <class T0, class T1>
typename std::enable_if<T1::IsVectorAtCompileTime, void>::type indirect_transform(
    Eigen::ArrayBase<T0> const &coeffs, Eigen::ArrayBase<T1> &signal, t_uint levels,
    WaveletData const &wavelet) {
  if (levels == 0) return;
  assert(coeffs.rows() == signal.rows());
  assert(coeffs.cols() == signal.cols());
  assert(coeffs.size() % (1u << levels) == 0);

  auto input = copy(coeffs);
  for (t_uint level(levels - 1); level > 0; --level) {
    auto const N = static_cast<t_uint>(signal.size()) >> level;
    indirect_transform_impl(input.head(N), signal.head(N), wavelet);
    input.head(N) = signal.head(N);
  }
  indirect_transform_impl(input, signal, wavelet);
}

//! \brief N-levels 2d indirect transform
//! \param[in] coeffs_: input coefficients
//! \param[out] signal: output with the reconstituted signal
//! \param[in] wavelet: contains wavelet coefficients
//! \note The size  of the signal and coefficients should a multiple of $2^l$ where $l$ is the
//! number of levels.
template <class T0, class T1>
typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type indirect_transform(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal_, t_uint levels,
    WaveletData const &wavelet) {
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);
  Eigen::ArrayBase<T1> &signal = const_cast<Eigen::ArrayBase<T1> &>(signal_);
  assert(coeffs.rows() == signal.rows());
  assert(coeffs.cols() == signal.cols());
  assert(coeffs.size() % (1u << levels) == 0);
  if (levels == 0) {
    signal = coeffs_;
    return;
  }

  auto input = copy(coeffs);
  for (t_uint level(levels - 1); level > 0; --level) {
    auto const Nx = static_cast<t_uint>(signal.rows()) >> level;
    auto const Ny = static_cast<t_uint>(signal.cols()) >> level;
    indirect_transform_impl(input.topLeftCorner(Nx, Ny), signal.topLeftCorner(Nx, Ny), wavelet);
    input.topLeftCorner(Nx, Ny) = signal.topLeftCorner(Nx, Ny);
  }
  indirect_transform_impl(input, signal, wavelet);
}
#ifdef SOPT_MPI
//! \brief N-levels 2d indirect transform
//! \param[in] coeffs_: input coefficients
//! \param[out] signal: output with the reconstituted signal
//! \param[in] wavelet: contains wavelet coefficients
//! \note The size  of the signal and coefficients should a multiple of $2^l$ where $l$ is the
//! number of levels.
template <class T0, class T1>
typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type indirect_transform(
    Eigen::ArrayBase<T0> const &coeffs_, Eigen::ArrayBase<T1> const &signal_, t_uint levels,
    WaveletData const &wavelet, const mpi::Communicator &comm) {
  Eigen::ArrayBase<T0> &coeffs = const_cast<Eigen::ArrayBase<T0> &>(coeffs_);
  Eigen::ArrayBase<T1> &signal = const_cast<Eigen::ArrayBase<T1> &>(signal_);
  assert(coeffs.rows() == signal.rows());
  assert(coeffs.cols() == signal.cols());
  assert(coeffs.size() % (1u << levels) == 0);
  if (levels == 0) {
    signal = coeffs_;
    return;
  }

  auto input = copy(coeffs);
  for (t_uint level(levels - 1); level > 0; --level) {
    auto const Nx = static_cast<t_uint>(signal.rows()) >> level;
    auto const Ny = static_cast<t_uint>(signal.cols()) >> level;
    indirect_transform_impl(
        input.topLeftCorner(Nx, Ny), signal.topLeftCorner(Nx, Ny), wavelet, comm);
    input.topLeftCorner(Nx, Ny) = signal.topLeftCorner(Nx, Ny);
  }
  indirect_transform_impl(input, signal, wavelet, comm);
}
//! Indirect 1d and 2d transform with MPI
//! \param[in] coeffs_: input coefficients
//! \param[in] wavelet: contains wavelet coefficients
//! \returns the reconstituted signal
//! \note The size  of the coefficients should a multiple of $2^l$ where $l$ is the number of
//! levels.
template <class T0>
auto indirect_transform(Eigen::ArrayBase<T0> const &coeffs, t_uint levels,
                        WaveletData const &wavelet, const mpi::Communicator &comm)
    -> decltype(copy(coeffs)) {
  auto result = copy(coeffs);
  indirect_transform(coeffs, result, levels, wavelet, comm);
  return result;
}
#endif

//! Indirect 1d and 2d transform
//! \param[in] coeffs_: input coefficients
//! \param[in] wavelet: contains wavelet coefficients
//! \returns the reconstituted signal
//! \note The size  of the coefficients should a multiple of $2^l$ where $l$ is the number of
//! levels.
template <class T0>
auto indirect_transform(Eigen::ArrayBase<T0> const &coeffs, t_uint levels,
                        WaveletData const &wavelet) -> decltype(copy(coeffs)) {
  auto result = copy(coeffs);
  indirect_transform(coeffs, result, levels, wavelet);
  return result;
}
}  // namespace wavelets
}  // namespace sopt
#endif
