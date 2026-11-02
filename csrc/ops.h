#pragma once

#include <torch/extension.h>

torch::Tensor ibcs_cuda(torch::Tensor q, torch::Tensor k);

void sip_cuda(torch::Tensor scores, torch::Tensor offsets,
              torch::Tensor ids, torch::Tensor logs);
