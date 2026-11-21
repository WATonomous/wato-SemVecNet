// This file is generated.  Do not edit manually.

// Dependent includes
#include <vdpau/vdpau.h>

// CUDA public interface, for type definitions and api function prototypes
#include "cuda_vdpau_interop.h"


// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct cudaVDPAUGetDevice_v3020_params_st {
    int *device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} cudaVDPAUGetDevice_v3020_params;

typedef struct cudaVDPAUSetVDPAUDevice_v3020_params_st {
    int device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} cudaVDPAUSetVDPAUDevice_v3020_params;

typedef struct cudaGraphicsVDPAURegisterVideoSurface_v3020_params_st {
    struct cudaGraphicsResource **resource;
    VdpVideoSurface vdpSurface;
    unsigned int flags;
} cudaGraphicsVDPAURegisterVideoSurface_v3020_params;

typedef struct cudaGraphicsVDPAURegisterOutputSurface_v3020_params_st {
    struct cudaGraphicsResource **resource;
    VdpOutputSurface vdpSurface;
    unsigned int flags;
} cudaGraphicsVDPAURegisterOutputSurface_v3020_params;
