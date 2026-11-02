#pragma once

#include <cstdint>
#include <torch/extension.h>
#include <vector>

torch::Tensor ibcs_cuda(torch::Tensor q, torch::Tensor k);

std::vector<torch::Tensor> sip_cuda(
    torch::Tensor scores, torch::Tensor offsets, int64_t samples);
