#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("ibcs", &ibcs_cuda,
               "IBCS: population-weighted block importance scores.");
    module.def("sip", &sip_cuda,
               "SIP: systematic sampling and inverse-inclusion log2 weights.");
}
