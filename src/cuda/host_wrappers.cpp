#include "host_trace.h"

#include <dlfcn.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>

#if defined(RATELPROF_GENERATED_HOST_WRAPPERS) && defined(RATELPROF_OMP_HEADER)
#include RATELPROF_OMP_HEADER
#endif

#ifdef RATELPROF_CUDA_CUBLAS
#include <cublas_v2.h>
#if __has_include(<cublasLt.h>)
#include <cublasLt.h>
#define RATELPROF_HAS_CUBLAS_LT
#endif
#endif

#ifdef RATELPROF_CUDA_NCCL
#include <nccl.h>
#endif

#ifdef RATELPROF_CUDA_MPI
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#endif

namespace {

template<class Function>
Function resolve(const char *name)
{
    auto fn = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
    if (!fn) {
        std::fprintf(stderr, "RATELProf CUDA: unable to resolve %s: %s\n", name, dlerror());
        std::abort();
    }
    return fn;
}

template<class Call, class ReadArguments>
auto traced(HostDomain domain, const char *name, const void *caller, Call call,
            ReadArguments read_arguments)
{
    using Result = decltype(call());
    TraceArguments arguments;
    if (host_arguments_enabled(domain))
        arguments = read_arguments();
    const uint64_t start = host_begin(domain);
    if constexpr (std::is_void_v<Result>) {
        call();
        if (start)
            host_end(domain, name, start, caller, int64_t{0}, false, std::move(arguments));
    } else {
        auto result = call();
        if (start)
            host_end(domain, name, start, caller, trace_argument_value(result), true, std::move(arguments));
        return result;
    }
}

} // namespace

#ifdef RATELPROF_GENERATED_HOST_WRAPPERS
#include RATELPROF_GENERATED_HOST_WRAPPERS
#else

#define WRAP(domain, result, name, params, args)                                   \
    extern "C" result name params                                                  \
    {                                                                             \
        static const auto original = resolve<result (*) params>(#name);            \
        return traced(domain, #name, __builtin_return_address(0),                   \
                      [&] { return original args; }, [] { return TraceArguments{}; });                              \
    }

#ifdef RATELPROF_CUDA_CUBLAS
WRAP(Cublas, cublasStatus_t, cublasCreate_v2, (cublasHandle_t *handle), (handle))
WRAP(Cublas, cublasStatus_t, cublasDestroy_v2, (cublasHandle_t handle), (handle))
WRAP(Cublas, cublasStatus_t, cublasSetStream_v2, (cublasHandle_t handle, cudaStream_t stream), (handle, stream))
WRAP(Cublas, cublasStatus_t, cublasSetPointerMode_v2,
     (cublasHandle_t handle, cublasPointerMode_t mode), (handle, mode))
WRAP(Cublas, cublasStatus_t, cublasSgemm_v2,
     (cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
      const float *alpha, const float *a, int lda, const float *b, int ldb,
      const float *beta, float *c, int ldc),
     (h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc))
WRAP(Cublas, cublasStatus_t, cublasDgemm_v2,
     (cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
      const double *alpha, const double *a, int lda, const double *b, int ldb,
      const double *beta, double *c, int ldc),
     (h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc))
#if CUDART_VERSION >= 11000
WRAP(Cublas, cublasStatus_t, cublasGemmEx,
     (cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
      const void *alpha, const void *a, cudaDataType at, int lda, const void *b, cudaDataType bt, int ldb,
      const void *beta, void *c, cudaDataType ct, int ldc, cublasComputeType_t compute, cublasGemmAlgo_t algo),
     (h, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc, compute, algo))
#endif
WRAP(Cublas, cublasStatus_t, cublasSgemmStridedBatched,
     (cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
      const float *alpha, const float *a, int lda, long long sa, const float *b, int ldb, long long sb,
      const float *beta, float *c, int ldc, long long sc, int batch),
     (h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, batch))
#ifdef RATELPROF_HAS_CUBLAS_LT
WRAP(Cublas, cublasStatus_t, cublasLtCreate, (cublasLtHandle_t *handle), (handle))
WRAP(Cublas, cublasStatus_t, cublasLtDestroy, (cublasLtHandle_t handle), (handle))
WRAP(Cublas, cublasStatus_t, cublasLtMatmul,
     (cublasLtHandle_t h, cublasLtMatmulDesc_t compute, const void *alpha, const void *a,
      cublasLtMatrixLayout_t ad, const void *b, cublasLtMatrixLayout_t bd, const void *beta,
      const void *c, cublasLtMatrixLayout_t cd, void *d, cublasLtMatrixLayout_t dd,
      const cublasLtMatmulAlgo_t *algo, void *workspace, size_t bytes, cudaStream_t stream),
     (h, compute, alpha, a, ad, b, bd, beta, c, cd, d, dd, algo, workspace, bytes, stream))
#endif
#endif

#ifdef RATELPROF_CUDA_NCCL
WRAP(Nccl, ncclResult_t, ncclCommInitRank, (ncclComm_t *comm, int count, ncclUniqueId id, int rank),
     (comm, count, id, rank))
WRAP(Nccl, ncclResult_t, ncclCommInitAll, (ncclComm_t *comms, int count, const int *devices),
     (comms, count, devices))
WRAP(Nccl, ncclResult_t, ncclCommDestroy, (ncclComm_t comm), (comm))
WRAP(Nccl, ncclResult_t, ncclGroupStart, (), ())
WRAP(Nccl, ncclResult_t, ncclGroupEnd, (), ())
WRAP(Nccl, ncclResult_t, ncclBroadcast,
     (const void *src, void *dst, size_t count, ncclDataType_t type, int root, ncclComm_t c, cudaStream_t s),
     (src, dst, count, type, root, c, s))
WRAP(Nccl, ncclResult_t, ncclAllGather,
     (const void *src, void *dst, size_t count, ncclDataType_t type, ncclComm_t c, cudaStream_t s),
     (src, dst, count, type, c, s))
WRAP(Nccl, ncclResult_t, ncclAllReduce,
     (const void *src, void *dst, size_t count, ncclDataType_t type, ncclRedOp_t op, ncclComm_t c, cudaStream_t s),
     (src, dst, count, type, op, c, s))
WRAP(Nccl, ncclResult_t, ncclReduceScatter,
     (const void *src, void *dst, size_t count, ncclDataType_t type, ncclRedOp_t op, ncclComm_t c, cudaStream_t s),
     (src, dst, count, type, op, c, s))
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >= 7)
WRAP(Nccl, ncclResult_t, ncclSend,
     (const void *src, size_t count, ncclDataType_t type, int peer, ncclComm_t c, cudaStream_t s),
     (src, count, type, peer, c, s))
WRAP(Nccl, ncclResult_t, ncclRecv,
     (void *dst, size_t count, ncclDataType_t type, int peer, ncclComm_t c, cudaStream_t s),
     (dst, count, type, peer, c, s))
#endif
#endif

#ifdef RATELPROF_CUDA_OMP_ROUTINES
WRAP(OmpRoutine, int, omp_get_num_devices, (), ())
WRAP(OmpRoutine, int, omp_get_initial_device, (), ())
WRAP(OmpRoutine, int, omp_get_default_device, (), ())
WRAP(OmpRoutine, void, omp_set_default_device, (int device), (device))
WRAP(OmpRoutine, void *, omp_target_alloc, (size_t bytes, int device), (bytes, device))
WRAP(OmpRoutine, void, omp_target_free, (void *p, int device), (p, device))
WRAP(OmpRoutine, int, omp_target_is_present, (const void *p, int device), (p, device))
WRAP(OmpRoutine, int, omp_target_memcpy,
     (void *dst, const void *src, size_t size, size_t dst_offset, size_t src_offset, int dst_dev, int src_dev),
     (dst, src, size, dst_offset, src_offset, dst_dev, src_dev))
WRAP(OmpRoutine, int, omp_target_memcpy_rect,
     (void *dst, const void *src, size_t element, int dims, const size_t *volume, const size_t *dst_offsets,
      const size_t *src_offsets, const size_t *dst_dims, const size_t *src_dims, int dst_dev, int src_dev),
     (dst, src, element, dims, volume, dst_offsets, src_offsets, dst_dims, src_dims, dst_dev, src_dev))
WRAP(OmpRoutine, int, omp_target_associate_ptr,
     (const void *host, const void *device, size_t size, size_t offset, int dev),
     (host, device, size, offset, dev))
WRAP(OmpRoutine, int, omp_target_disassociate_ptr, (const void *host, int dev), (host, dev))
#endif

#ifdef RATELPROF_CUDA_MPI
WRAP(Mpi, int, MPI_Init, (int *argc, char ***argv), (argc, argv))
WRAP(Mpi, int, MPI_Init_thread, (int *argc, char ***argv, int required, int *provided),
     (argc, argv, required, provided))
WRAP(Mpi, int, MPI_Finalize, (), ())
WRAP(Mpi, int, MPI_Initialized, (int *flag), (flag))
WRAP(Mpi, int, MPI_Comm_rank, (MPI_Comm comm, int *rank), (comm, rank))
WRAP(Mpi, int, MPI_Comm_size, (MPI_Comm comm, int *size), (comm, size))
WRAP(Mpi, int, MPI_Barrier, (MPI_Comm comm), (comm))
WRAP(Mpi, int, MPI_Ibarrier, (MPI_Comm comm, MPI_Request *request), (comm, request))
WRAP(Mpi, int, MPI_Bcast, (void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm),
     (buf, count, type, root, comm))
WRAP(Mpi, int, MPI_Ibcast,
     (void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm, MPI_Request *request),
     (buf, count, type, root, comm, request))
WRAP(Mpi, int, MPI_Send, (const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm),
     (buf, count, type, dest, tag, comm))
WRAP(Mpi, int, MPI_Recv,
     (void *buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm, MPI_Status *status),
     (buf, count, type, source, tag, comm, status))
WRAP(Mpi, int, MPI_Sendrecv,
     (const void *send, int sc, MPI_Datatype st, int dest, int stag, void *recv, int rc, MPI_Datatype rt,
      int source, int rtag, MPI_Comm comm, MPI_Status *status),
     (send, sc, st, dest, stag, recv, rc, rt, source, rtag, comm, status))
WRAP(Mpi, int, MPI_Isend,
     (const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm, MPI_Request *request),
     (buf, count, type, dest, tag, comm, request))
WRAP(Mpi, int, MPI_Irecv,
     (void *buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm, MPI_Request *request),
     (buf, count, type, source, tag, comm, request))
WRAP(Mpi, int, MPI_Wait, (MPI_Request *request, MPI_Status *status), (request, status))
WRAP(Mpi, int, MPI_Waitall, (int count, MPI_Request *requests, MPI_Status *statuses), (count, requests, statuses))
WRAP(Mpi, int, MPI_Waitany, (int count, MPI_Request *requests, int *index, MPI_Status *status),
     (count, requests, index, status))
WRAP(Mpi, int, MPI_Test, (MPI_Request *request, int *flag, MPI_Status *status), (request, flag, status))
WRAP(Mpi, int, MPI_Testall, (int count, MPI_Request *requests, int *flag, MPI_Status *statuses),
     (count, requests, flag, statuses))
WRAP(Mpi, int, MPI_Reduce,
     (const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm),
     (send, recv, count, type, op, root, comm))
WRAP(Mpi, int, MPI_Allreduce,
     (const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, MPI_Comm comm),
     (send, recv, count, type, op, comm))
WRAP(Mpi, int, MPI_Iallreduce,
     (const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, MPI_Comm comm, MPI_Request *request),
     (send, recv, count, type, op, comm, request))
WRAP(Mpi, int, MPI_Gather,
     (const void *send, int sc, MPI_Datatype st, void *recv, int rc, MPI_Datatype rt, int root, MPI_Comm comm),
     (send, sc, st, recv, rc, rt, root, comm))
WRAP(Mpi, int, MPI_Allgather,
     (const void *send, int sc, MPI_Datatype st, void *recv, int rc, MPI_Datatype rt, MPI_Comm comm),
     (send, sc, st, recv, rc, rt, comm))
WRAP(Mpi, int, MPI_Scatter,
     (const void *send, int sc, MPI_Datatype st, void *recv, int rc, MPI_Datatype rt, int root, MPI_Comm comm),
     (send, sc, st, recv, rc, rt, root, comm))
WRAP(Mpi, int, MPI_Alltoall,
     (const void *send, int sc, MPI_Datatype st, void *recv, int rc, MPI_Datatype rt, MPI_Comm comm),
     (send, sc, st, recv, rc, rt, comm))
#endif

#endif // RATELPROF_GENERATED_HOST_WRAPPERS
