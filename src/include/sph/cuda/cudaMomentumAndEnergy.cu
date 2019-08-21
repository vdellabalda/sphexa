#include <cuda.h>

#include "cudaMomentumAndEnergy.cuh"
#include "../kernels.hpp"

namespace sphexa
{
namespace sph
{

template void cudaComputeMomentumAndEnergy<double, SqPatch<double>>(const std::vector<int> &clist, SqPatch<double> &d);

const double gradh_i = 1.0;
const double gradh_j = 1.0;
const double ep1 = 0.2, ep2 = 0.02;
const int mre = 4;

#define CHECK_CUDA_ERR(errcode) checkErr((errcode), __FILE__, __LINE__, #errcode);

void checkErr(cudaError_t err, const char *filename, int lineno, const char *funcName)
{
    if (err != cudaSuccess)
    {
        const char *errName = cudaGetErrorName(err);
        const char *errStr = cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error at %s:%d. Function %s returned err %d: %s - %s\n", filename, lineno, funcName, err, errName, errStr);
    }
}

template <typename T>
__global__ void momenumAndEnergy_manyParticlesPerBlock(const int n, const int dx, const T sincIndex, const T K, const int ngmax,
                                                       const BBox<T> *bbox, const int *clist, const int *neighbors,
                                                       const int *neighborsCount, const T *x, const T *y, const T *z, const T *vx,
                                                       const T *vy, const T *vz, const T *h, const T *m, const T *ro, const T *p,
                                                       const T *c, T *grad_P_x, T *grad_P_y, T *grad_P_z, T *du)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    const int i = clist[tid];
    const int nn = neighborsCount[tid];
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;

    T A_i = 0.0;
    if (p[i] < 0.0) A_i = 1.0;

    // int converstion to avoid a bug that prevents vectorization with some compilers
    for (int pj = 0; pj < nn; pj++)
    {
        const int j = neighbors[tid * ngmax + pj];

        // calculate the scalar product rv = rij * vij
        T r_ijx = (x[i] - x[j]);
        T r_ijy = (y[i] - y[j]);
        T r_ijz = (z[i] - z[j]);

        applyPBC(*bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);

        const T v_ijx = (vx[i] - vx[j]);
        const T v_ijy = (vy[i] - vy[j]);
        const T v_ijz = (vz[i] - vz[j]);

        const T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

        const T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

        const T r_ij = sqrt(r_square);
        const T rv_i = r_ij / h[i];
        const T rv_j = r_ij / h[j];
        const T viscosity_ij = artificial_viscosity(ro[i], ro[j], h[i], h[j], c[i], c[j], rv, r_square);

        const T derivative_kernel_i = wharmonic_derivative(rv_i, h[i], sincIndex, K);
        const T derivative_kernel_j = wharmonic_derivative(rv_j, h[j], sincIndex, K);

        // divide by r_ij? missing h?
        const T grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
        const T grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
        const T grad_v_kernel_z_i = r_ijz * derivative_kernel_i;

        const T grad_v_kernel_x_j = r_ijx * derivative_kernel_j;
        const T grad_v_kernel_y_j = r_ijy * derivative_kernel_j;
        const T grad_v_kernel_z_j = r_ijz * derivative_kernel_j;

        const T grad_v_kernel_x_ij = (grad_v_kernel_x_i + grad_v_kernel_x_j) / 2.0;
        const T grad_v_kernel_y_ij = (grad_v_kernel_y_i + grad_v_kernel_y_j) / 2.0;
        const T grad_v_kernel_z_ij = (grad_v_kernel_z_i + grad_v_kernel_z_j) / 2.0;

        const T force_i_j_r = exp(-(rv_i * rv_i)) * exp((dx * dx) / (h[i] * h[i]));

        T A_j = 0.0;
        if (p[j] < 0.0) A_j = 1.0;

        T delta_pos_i_j = 0.0;
        if (p[i] > 0.0 && p[j] > 0.0) delta_pos_i_j = 1.0;

        const T R_i_j = ep1 * (A_i * abs(p[i]) + A_j * abs(p[j])) + ep2 * delta_pos_i_j * (abs(p[i]) + abs(p[j]));

        const T r_force_i_j = R_i_j * pow(force_i_j_r, (int)mre);

        const T partial_repulsive_force = (r_force_i_j / (ro[i] * ro[j]));

        const T pro_i = p[i] / (gradh_i * ro[i] * ro[i]);
        const T pro_j = p[j] / (gradh_j * ro[j] * ro[j]);

        momentum_x +=
            m[j] * (pro_i * grad_v_kernel_x_i + pro_j * grad_v_kernel_x_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_x_ij);
        momentum_y +=
            m[j] * (pro_i * grad_v_kernel_y_i + pro_j * grad_v_kernel_y_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_y_ij);
        momentum_z +=
            m[j] * (pro_i * grad_v_kernel_z_i + pro_j * grad_v_kernel_z_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_z_ij);

        energy += m[j] * (pro_i + 0.5 * viscosity_ij) * (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
    }

    du[tid] = energy;

    grad_P_x[tid] = momentum_x;
    grad_P_y[tid] = momentum_y;
    grad_P_z[tid] = momentum_z;
}

template <typename T, class Dataset>
void cudaComputeMomentumAndEnergy(const std::vector<int> &clist, Dataset &d)
{
    const size_t n = clist.size();
    const size_t ngmax = d.ngmax;
    const T dx = d.dx;
    const T sincIndex = d.sincIndex;
    const T K = d.K;
    const BBox<T> bbox = d.bbox;
    const size_t np = d.x.size();

    int *d_clist, *d_neighbors, *d_neighborsCount; // d_ prefix stands for device
    T *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_h, *d_m, *d_ro, *d_p, *d_c;
    T *d_grad_P_x, *d_grad_P_y, *d_grad_P_z, *d_du;
    BBox<T> *d_bbox;

    const size_t n_chunk = (size_t)n / d.noOfGpuLoopSplits;
    const size_t allNeighbors_chunk = n_chunk * ngmax;

    const size_t size_allNeighbors_chunk = allNeighbors_chunk * sizeof(int);
    const size_t size_n_T_slice = n_chunk * sizeof(T);
    const size_t size_n_int_slice = n_chunk * sizeof(int);

    const size_t size_np_T = np * sizeof(T);
    const size_t size_bbox = sizeof(BBox<T>);

    // const float neighborsSizeInGB = size_allNeighbors_chunk * 1e-9;
    // const float memorySizeInGB = (2 * size_n_int_slice + size_allNeighbors_chunk + size_bbox + 11 * size_np_T_slice + 4 *
    // size_n_T_slice)*1e-9; printf("CUDA: Total GPU memory usage: %.2fGB\n", memorySizeInGB);

    // input data
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_clist, size_n_int_slice));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_neighbors, size_allNeighbors_chunk));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_neighborsCount, size_n_int_slice));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_bbox, size_bbox));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_x, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_y, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_z, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_vx, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_vy, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_vz, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_h, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_m, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_ro, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_p, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_c, size_np_T));

    // output data
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_grad_P_x, size_n_T_slice));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_grad_P_y, size_n_T_slice));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_grad_P_z, size_n_T_slice));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_du, size_n_T_slice));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_vx, d.vx.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_vy, d.vy.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_vz, d.vz.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_p, d.p.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_c, d.c.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &bbox, size_bbox, cudaMemcpyHostToDevice));

    for (ushort s = 0; s < d.noOfGpuLoopSplits; ++s)
    {
        CHECK_CUDA_ERR(cudaMemcpy(d_clist, clist.data() + (s * n_chunk), size_n_int_slice, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(
            cudaMemcpy(d_neighbors, d.neighbors.data() + (s * allNeighbors_chunk), size_allNeighbors_chunk, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_neighborsCount, d.neighborsCount.data() + (s * n_chunk), size_n_int_slice, cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (n / d.noOfGpuLoopSplits + threadsPerBlock - 1) / threadsPerBlock;
        // printf("CUDA kernel [manyParticlesPerBlock] launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
        momenumAndEnergy_manyParticlesPerBlock<T><<<blocksPerGrid, threadsPerBlock>>>(
            n / d.noOfGpuLoopSplits, dx, sincIndex, K, ngmax, d_bbox, d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_vx, d_vy,
            d_vz, d_h, d_m, d_ro, d_p, d_c, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du);

        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data() + (s * n_chunk), d_grad_P_x, size_n_T_slice, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data() + (s * n_chunk), d_grad_P_y, size_n_T_slice, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data() + (s * n_chunk), d_grad_P_z, size_n_T_slice, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.du.data() + (s * n_chunk), d_du, size_n_T_slice, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA_ERR(cudaFree(d_clist));
    CHECK_CUDA_ERR(cudaFree(d_neighborsCount));
    CHECK_CUDA_ERR(cudaFree(d_neighbors));

    CHECK_CUDA_ERR(cudaFree(d_x));
    CHECK_CUDA_ERR(cudaFree(d_y));
    CHECK_CUDA_ERR(cudaFree(d_z));
    CHECK_CUDA_ERR(cudaFree(d_vx));
    CHECK_CUDA_ERR(cudaFree(d_vy));
    CHECK_CUDA_ERR(cudaFree(d_vz));
    CHECK_CUDA_ERR(cudaFree(d_h));
    CHECK_CUDA_ERR(cudaFree(d_m));
    CHECK_CUDA_ERR(cudaFree(d_ro));
    CHECK_CUDA_ERR(cudaFree(d_p));
    CHECK_CUDA_ERR(cudaFree(d_c));

    CHECK_CUDA_ERR(cudaFree(d_grad_P_x));
    CHECK_CUDA_ERR(cudaFree(d_grad_P_y));
    CHECK_CUDA_ERR(cudaFree(d_grad_P_z));
    CHECK_CUDA_ERR(cudaFree(d_du));
}
} // namespace sph
} // namespace sphexa