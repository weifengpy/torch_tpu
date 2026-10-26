/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Registers the nonzero indices kernel with the unbounded dynamic kernel
// registry when this library is loaded. Nothing references this file, so it is
// built with `alwayslink = True`; link it only into TPU-only targets.

#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

namespace torch_tpu {
namespace {

// Registers a kernel with `UnboundedDynamicKernelRegistry` when constructed.
// Prefer `TT_REGISTER_DYNAMIC_KERNEL` over instantiating this class directly.
class UnboundedDynamicKernelRegistrar {
 public:
  UnboundedDynamicKernelRegistrar(ShapelessKey key,
                                  UnboundedDynamicKernel::BuildFn
                                  absl_nonnull build_fn,
                                  UnboundedDynamicExecutionMetadata metadata,
                                  std::string id) {
    CompilationCacheKey cache_key = CompilationCacheKey(
        GraphKey(key, DimensionsKey({})), CompileOptionsKey({}), std::move(id));
    ABSL_CHECK_OK(  // CRASH_OK=static-init registration bug
        UnboundedDynamicKernelRegistry::GetInstance().RegisterKernel(

            UnboundedDynamicKernel{.build = build_fn,
                                   .metadata = std::move(metadata),
                                   .key = std::move(cache_key)}));
  }
};

#define TT_REGISTER_DYNAMIC_KERNEL_CONCAT_INNER_(x, y) x##y
#define TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(x, y) \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_INNER_(x, y)

// Registers an unbounded dynamic kernel with `UnboundedDynamicKernelRegistry`
// at static initialization time. Invoke at namespace scope in a library built
// with `alwayslink = True`:
//
//   TT_REGISTER_DYNAMIC_KERNEL(NonZeroCacheKey(),
//                              BuildNonzeroIndicesExecutable,
//                              MakeNonzeroDynamicMetadata(), "nonzero_2d");
//
// `key` is the shapeless key of the graphs the kernel serves. The kernel is
// registered under the compilation cache key formed from `key`, empty
// dimensions and compile options keys, and `id`, which tells apart kernels
// that share a shapeless key.
//
// The arguments are evaluated during static initialization, so they must not
// depend on dynamically initialized globals from other translation units
// (flags, environment-derived settings, other registries, ...).
//
// Registering the same shapeless key and id twice is a programming error and
// crashes at startup.
#define TT_REGISTER_DYNAMIC_KERNEL(key, build_fn, metadata, id)         \
  const ::torch_tpu::UnboundedDynamicKernelRegistrar                    \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(kUnboundedDynamicKernelRegistrar_, \
                                     __COUNTER__)((key), (build_fn),    \
                                                  (metadata), (id))

TT_REGISTER_DYNAMIC_KERNEL(NonZeroCacheKey(), BuildNonzeroIndicesExecutable,
                           MakeNonzeroDynamicMetadata(), "nonzero_2d");

}  // namespace
}  // namespace torch_tpu
