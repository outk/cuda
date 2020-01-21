#include <stdio.h>

__global__ void helloFromGPU()
{
    unsigned int a = threadIdx.x;

    if (a == 5)
    {
        printf("Hello World from GPU thread %d!\n", a);
    }
}

int main(int argc, char **argv)
{
    printf("Hello world from CPU!\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    // cudaDeviceSynchronize();
    return 0;
}