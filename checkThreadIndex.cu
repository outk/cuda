#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) {                                                       \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess) {                                             \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                            \
    }                                                                       \
}                                                                           

void printMatrix(int *C, const int nx, const int ny) {
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);
    for (int ix = 0; ix < nx; ix++) {
        printf("%3d", ic[ix]);
    }
    ic += nx;
    printf("\n");
    return;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv) {
    printf("%s Starting ... \n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 8;
    int ny = 6;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);

    int *h_A;
    h_A = (int *)malloc(nBytes);

    for (int i = 0; i < nxy; i++) {
        h_A[i] = i;
    }
    printMatrix(h_A, nx, ny);

    int *d_MatA;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));

    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(d_MatA));
    free(h_A);

    CHECK(cudaDeviceReset());

    return(0);
}