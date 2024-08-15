#include "cuarena.cuh"

struct dummy
{
    int a;
    int b;
    int c;
    int d;
};

int main(int argc, char *argv[])
{
    CUcontext ctx;
    CUdevice dev;

    CU_CHECK(cuInit(0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, 0));
    CU_CHECK(cuCtxSetCurrent(ctx));
    CU_CHECK(cuCtxGetDevice(&dev));

    int supports_virtual_memory = 0;
    CU_CHECK(cuDeviceGetAttribute(&supports_virtual_memory, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
    if (supports_virtual_memory) {
        printf("Virtual memory support detected\n");
    }

    printf("Context: %p, Device: %d\n", ctx, dev);

    auto n_bytes = 2_GB;
    arena a(ctx);
    memblk buffer = a.allocate(n_bytes);
    a.deallocate(buffer);
    memblk buffer2 = a.allocate(n_bytes);
    memblk buffer3 = a.allocate(n_bytes);
    memblk buffer4 = a.allocate(n_bytes);

    // pool p(ctx, { .slot_size = sizeof(dummy), .capacity = 4 * sizeof(dummy), .alignment = 16 });
    //
    // memblk b1 = p.allocate();
    // memblk b2 = p.allocate();
    // memblk b3 = p.allocate();
    // p.deallocate(b3);
    // memblk b4 = p.allocate();
    // memblk b5 = p.allocate();
    // memblk b6 = p.allocate();
    // p.clear();

    CU_CHECK(cuDevicePrimaryCtxRelease(0));

    return 0;
}
