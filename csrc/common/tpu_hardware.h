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
#ifndef TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_
#define TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace torch_tpu {

inline constexpr std::string_view kDefaultPciDevicesDir =
    "/sys/bus/pci/devices";
inline constexpr std::string_view kDefaultVfioDir = "/dev/vfio";

using TpuTopology = std::vector<int64_t>;  // INT_VEC_OK
using TpuTopologyMap = absl::flat_hash_map<int64_t, TpuTopology>;

// SparseCore specs of a TPU generation.
//
// SparseCores are TPU processors specialized for sparse operations such as
// embedding lookups.
struct SparseCoreSpec {
  // Number of SparseCores per TPU device. On generations that expose each
  // TensorCore of a chip as a separate device (e.g. TPU v7), this counts the
  // SparseCores of one device, not of the chip.
  int64_t sparse_cores_per_device = 0;
  // SIMD width (number of vector lanes) of each SparseCore.
  int64_t simd_width = 0;

  bool operator==(const SparseCoreSpec& other) const = default;
};

// Specs of a TPU generation that TorchTPU supports.
//
// TorchTPU doesn't run on TPU v4 and older generations, so they have no
// constants on `TpuSpec`.
struct TpuSpec {
  // Human-readable name of the generation, as returned by
  // `torch.tpu.get_device_properties().name`.
  std::string name;
  // Number of separately-addressable devices the chip exposes to the runtime,
  // used as the divisor that turns a per-chip quantity into what one device
  // sees. This is megacore-aware (not the physical TensorCore count): 1 for
  // single-core chips (v5e, v6e, v8t) and megacore chips (v5p) that fuse their
  // two physical cores into one addressable device, and 2 for dual-core chips
  // without megacore (v7, v8i).
  int64_t devices_per_chip = 1;
  // Physical HBM capacity per chip, in bytes.
  int64_t hbm_bytes_per_chip = 0;
  // Peak dense bf16 compute per chip, in FLOP/s.
  int64_t bf16_flops_per_chip = 0;
  // Topology dimensions of each number of devices that a host can use.
  TpuTopologyMap topologies;
  // SparseCore specs of the generation, or std::nullopt if it has no
  // SparseCores.
  std::optional<SparseCoreSpec> sparse_core_spec;

  // Physical HBM capacity, in bytes, visible to a single device.
  [[nodiscard]] int64_t hbm_bytes_per_device() const {
    return hbm_bytes_per_chip / devices_per_chip;
  }

  // Peak dense bf16 compute, in FLOP/s, for a single device.
  [[nodiscard]] int64_t bf16_flops_per_device() const {
    return bf16_flops_per_chip / devices_per_chip;
  }

  bool operator==(const TpuSpec& other) const = default;

  static const TpuSpec& V5p();
  static const TpuSpec& V5e();
  static const TpuSpec& V6e();
  static const TpuSpec& V7();
  static const TpuSpec& V8i();
  static const TpuSpec& V8t();
};

// Returns the mapping from PCI device ID (e.g. "0x0076") to the corresponding
// `TpuSpec`.
[[nodiscard]] const absl::flat_hash_map<std::string, const TpuSpec*>&
GetTpuSpecByPciDeviceIdMap();

// Result of scanning PCI devices for VFIO-exposed TPUs.
struct PciTpuScanResult {
  int64_t device_count = 0;
  const TpuSpec* spec = nullptr;

  bool operator==(const PciTpuScanResult& other) const = default;
};

// Returns the `TpuSpec` of the attached TPU device by scanning PCI devices
// under `pci_devices_dir` for a Google TPU, or nullptr if no supported TPU is
// found. Does not require VFIO exposure and does not initialize any TPU
// runtime.
[[nodiscard]] const TpuSpec* GetTpuSpec(
    std::string_view pci_devices_dir = kDefaultPciDevicesDir);

// Scans PCI devices under `pci_devices_dir` for supported Google TPUs whose
// IOMMU group is exposed under `vfio_dir`. Returns the count of VFIO-exposed
// TPU devices and the `TpuSpec` of the first one found (or nullptr if none).
[[nodiscard]] PciTpuScanResult ScanPciTpus(
    std::string_view pci_devices_dir = kDefaultPciDevicesDir,
    std::string_view vfio_dir = kDefaultVfioDir);

// Returns whether any device node through which libtpu could reach a TPU
// exists:
// - a VFIO group node `<dev_dir>/vfio/<number>` (not the container node
//   `<dev_dir>/vfio/vfio`);
// - a VFIO per-device (iommufd) node, i.e. any entry of
//   `<dev_dir>/vfio/devices/`, e.g. `vfio0`;
// - an accel node `<dev_dir>/accel*`, `<dev_borg_dir>/accel*`,
//   `<dev_dir>/accel/*` or `<dev_borg_dir>/accel/*` (the layout of the kernel's
//   compute accelerator subsystem, e.g. /dev/accel/accel0) that is a character
//   device whose major is registered in `proc_devices` (i.e. backed by a
//   driver).
// This does not depend on PCI device IDs, so it also covers unknown TPU
// generations. Returns true if a directory exists
// but cannot be listed, or if an accel character device exists but
// `proc_devices` cannot be read, since the absence of TPU device nodes cannot
// be established then. Does not initialize the TPU runtime.
//
// A process without any of these nodes cannot use a TPU through the default
// device paths. Hosts with other VFIO or accel devices (e.g. a NIC bound to
// VFIO) also yield true, so true does not mean that a TPU is present.
[[nodiscard]] bool HasTpuDeviceNodes(
    std::string_view dev_dir = "/dev",
    std::string_view dev_borg_dir = "/dev_borg",
    std::string_view proc_devices = "/proc/devices");

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_
