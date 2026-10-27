# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for hardware discovery and configuration.

This is for reading from the pcie devs, and then from that, determining the
generation of TPU that we have (and that we have a TPU at all). If the user
decides to call the wrapper to generate our two additional environment
variables, then we use this to decode their topology given the generation of the
device and the world size.

`TpuSpec` holds the specs of each TPU generation that TorchTPU supports, and
`get_tpu_spec()` returns the specs of the attached TPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
import os
import pathlib
from typing import Final

import immutabledict
from torch_tpu._internal.utils import _hardware_backend
from torch_tpu._internal.utils import annotations

# Google's PCI vendor ID (found in /sys/bus/pci/devices/*/vendor)
_GOOGLE_PCI_VENDOR_ID: Final[str] = "0x1ae0"

# Directory with an entry per PCI device of the host.
_PCI_DEVICES_DIR: Final[pathlib.Path] = pathlib.Path("/sys/bus/pci/devices")

# Directory with a node per IOMMU group that VFIO exposes to user space.
_VFIO_DIR: Final[pathlib.Path] = pathlib.Path("/dev/vfio")

# GPU device names
_NVIDIA_GPU_DEVICES: Final[Set[str]] = frozenset([
    "/dev/nvidia0",
    "/dev/nvidiactl",  # Docker/Kubernetes
    "/dev/dxg",  # WSL2
])

SparseCoreSpec = _hardware_backend.SparseCoreSpec
TpuSpec = _hardware_backend.TpuSpec

# TPU topology maps represent the logical mesh of TPU chips as (X, Y, Z)
# dimensions. For example, (2, 2, 1) indicates a 2x2x1 mesh of 4 chips.
# The keys are the number of TPU devices, and the values are the topology
# dimensions. Generations that expose each TensorCore of a chip as a separate
# device (TPU v7 and TPU v8i) have a trailing cores-per-chip dimension, e.g.
# (1, 2, 1, 2) for the 4 devices of a 1x2x1 mesh of 2 chips.
_V5P_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = TpuSpec.V5P.topologies
_V5E_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = TpuSpec.V5E.topologies
_V6E_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = TpuSpec.V6E.topologies
_V7_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = TpuSpec.V7.topologies
_V8I_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = TpuSpec.V8I.topologies
_V8T_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = TpuSpec.V8T.topologies

# The `TpuSpec` of each TPU PCI device ID. Verified IDs from
# util/platforminfo/pci_ids.h.
_TPU_SPEC_BY_PCI_DEVICE_ID: Final[Mapping[str, TpuSpec]] = (
    immutabledict.immutabledict(
        _hardware_backend.get_tpu_spec_by_pci_device_id_map()
    )
)


def get_tpu_spec() -> TpuSpec | None:
  """Returns the specs of the attached TPU device.

  Scans PCI devices for a Google TPU to identify its generation. Does not
  initialize any TPU runtime — safe to call at import time and from concurrent
  processes.

  Returns:
    The `TpuSpec` of the attached TPU device, or None if no TPU generation that
    TorchTPU supports is attached.
  """
  return _hardware_backend.get_tpu_spec(os.fspath(_PCI_DEVICES_DIR))


@annotations.deprecated(
    version="0.2.1",
    reason="Use torch.tpu.get_device_properties().name instead.",
)
def get_tpu_device_name() -> str:
  """Returns a human-readable name for the attached TPU device.

  Deprecated: code outside TorchTPU should use
  `torch.tpu.get_device_properties().name` instead, which raises a RuntimeError
  rather than returning "TPU" if no TPU generation that TorchTPU supports is
  attached.

  Like `get_tpu_spec()`, does not initialize any TPU runtime.

  Returns:
    The `TpuSpec.name` of the attached TPU device (e.g., "TPU v7"), or "TPU" if
    no TPU generation that TorchTPU supports is attached.
  """
  tpu_spec = get_tpu_spec()
  return "TPU" if tpu_spec is None else tpu_spec.name


@annotations.deprecated(
    version="0.2.1",
    reason=(
        "Use torch.tpu.get_device_properties().hbm_bytes_per_device instead."
    ),
)
def get_hbm_bytes_per_device() -> int | None:
  """Returns the physical HBM capacity, in bytes, of one attached TPU device.

  Deprecated: code outside TorchTPU should use
  `torch.tpu.get_device_properties().hbm_bytes_per_device` instead, which raises
  a RuntimeError rather than returning None if no TPU generation that TorchTPU
  supports is attached.

  Returns:
    The `TpuSpec.hbm_bytes_per_device` of the attached TPU device, or None if no
    TPU generation that TorchTPU supports is attached.
  """
  tpu_spec = get_tpu_spec()
  return None if tpu_spec is None else tpu_spec.hbm_bytes_per_device


def _scan_pci_tpus() -> tuple[int, TpuSpec | None]:
  """Scans PCI bus for TPU devices.

  Returns:
    A tuple of (count, tpu_spec). `count` is the number of TPU devices exposed
    via VFIO. `tpu_spec` is the `TpuSpec` of the first of them, or None if VFIO
    exposes no TPU device.
  """
  return _hardware_backend.scan_pci_tpus(
      os.fspath(_PCI_DEVICES_DIR), os.fspath(_VFIO_DIR)
  )


@annotations.deprecated(
    version="0.2.1", reason="Use torch.accelerator.device_count() instead."
)
def get_tpu_device_count() -> int:
  """Returns how many physical TPU PCI devices are attached to this host.

  Does not initialize the PJRT runtime. Counts the PCI devices in
  `/sys/bus/pci/devices` with Google's vendor ID and a TPU device ID in
  `_TPU_SPEC_BY_PCI_DEVICE_ID` whose IOMMU group is exposed under `/dev/vfio`.
  Devices whose sysfs entries can't be read are skipped, and if
  `/sys/bus/pci/devices` can't be read at all, this returns 0. The count does
  NOT include:

  - TPUs whose PCI device ID isn't in `_TPU_SPEC_BY_PCI_DEVICE_ID`, e.g. future
    generations;
  - TPUs whose IOMMU group isn't exposed under `/dev/vfio`, which is normal in
    containers that expose only the VFIO groups allocated to them;
  - TPUs bound to a non-VFIO driver, e.g. reached through `/dev/accel*`.

  So 0 doesn't mean the host has no TPUs. It's also not the number of devices
  the PJRT runtime will use, which can be smaller.

  Deprecated: code outside TorchTPU should call
  `torch.accelerator.device_count()` instead, which returns this count until the
  TPU runtime is initialized, and the runtime's device count afterwards.

  Returns:
    The integer number of TPU devices detected.
  """
  count, _ = _scan_pci_tpus()
  return count


@annotations.deprecated(
    version="0.2.1",
    reason="Use torch.tpu.get_device_properties().topology() instead.",
)
def get_tpu_topology(world_size: int | None = None) -> str | None:
  """Returns the topology string of the attached TPU devices.

  Deprecated: code outside TorchTPU should call
  `torch.tpu.get_device_properties().topology()` instead, which raises a
  RuntimeError rather than returning None if no TPU generation that TorchTPU
  supports is attached.

  Args:
    world_size: If provided, use this as the target device count for topology
      lookup instead of the auto-detected count. Allows sub-slicing only on
      single-host, in supported topologies.

  Returns:
    A topology string like "2,2,1", or None if VFIO exposes no TPU device.

  Raises:
    RuntimeError: If a TPU device is detected but no topology is found for the
      `target_count` (either `world_size` or the auto-detected count).
  """
  count, tpu_spec = _scan_pci_tpus()
  if tpu_spec is None:
    return None
  topology = tpu_spec.topology(count if world_size is None else world_size)
  return ",".join(map(str, topology))


def has_nvidia_gpu() -> bool:
  """True if there's a visible nvidia gpu available on device, False otherwise."""

  return any(os.path.exists(d) for d in _NVIDIA_GPU_DEVICES)
