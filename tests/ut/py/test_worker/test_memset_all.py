# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from simpler.task_interface import WorkerType
from simpler.worker import (
    Worker,
    _CTRL_MEMSET,
    _Lifecycle,
    _decode_memset_payload,
    _encode_memset_payload,
)


def _initialized_worker(device_ids=(8, 9), platform="a2a3"):
    worker = Worker.__new__(Worker)
    worker._worker = MagicMock()
    worker._lifecycle = _Lifecycle.READY
    worker._config = {"device_ids": list(device_ids), "platform": platform}
    worker._py_control_timeout_s = 30.0
    worker._child_prov_lock = MagicMock()
    worker._child_prov_require_live_range = MagicMock()
    return worker


def test_memset_payload_round_trip():
    ranges = {9: (0x20000, 4096), 8: (0x10000, 8192)}
    decoded = _decode_memset_payload(_encode_memset_payload(ranges))
    assert decoded == {8: (0x10000, 8192), 9: (0x20000, 4096)}


def test_memset_all_maps_worker_ids_to_device_ids():
    worker = _initialized_worker()

    def broadcast(worker_type, sub_cmd, payload, digest, *, timeout_s):
        assert worker_type == WorkerType.NEXT_LEVEL
        assert sub_cmd == _CTRL_MEMSET
        assert digest is None
        assert timeout_s == 30.0
        # worker 0 -> device 8, worker 1 -> device 9
        assert _decode_memset_payload(payload) == {8: (0x1000, 64), 9: (0x2000, 128)}
        return [SimpleNamespace(ok=True, worker_type="NEXT_LEVEL", worker_id=0, error_message="")]

    worker._worker.broadcast_control_all.side_effect = broadcast
    worker.memset_all({0: (0x1000, 64), 1: (0x2000, 128)})
    worker._child_prov_require_live_range.assert_any_call(0, 0x1000, 64, api="memset_all")
    worker._child_prov_require_live_range.assert_any_call(1, 0x2000, 128, api="memset_all")


def test_memset_all_rejects_unknown_worker_id():
    worker = _initialized_worker()
    with pytest.raises(ValueError, match="out of range"):
        worker.memset_all({7: (0x1000, 64)})
    worker._worker.broadcast_control_all.assert_not_called()


def test_memset_all_skips_empty_range_set():
    worker = _initialized_worker()
    worker.memset_all({})
    worker._worker.broadcast_control_all.assert_not_called()


def test_memset_all_requires_initialized_worker():
    worker = _initialized_worker()
    worker._lifecycle = _Lifecycle.CLOSED
    with pytest.raises(RuntimeError, match=r"Worker\.init"):
        worker.memset_all({0: (0x1000, 64)})


def test_memset_all_propagates_child_failure():
    worker = _initialized_worker()
    worker._worker.broadcast_control_all.return_value = [
        SimpleNamespace(ok=False, worker_type="NEXT_LEVEL", worker_id=1, error_message="aclrtMemset rc=1")
    ]
    with pytest.raises(RuntimeError, match="memset_all failed"):
        worker.memset_all({0: (0x1000, 64)})


def test_device_memset_unavailable_on_sim():
    worker = _initialized_worker(platform="a2a3sim")
    assert worker.device_memset_available is False
    with pytest.raises(RuntimeError, match="no ACL runtime"):
        worker.memset_all({0: (0x1000, 64)})
    worker._worker.broadcast_control_all.assert_not_called()


def test_device_memset_available_on_onboard():
    assert _initialized_worker(platform="a2a3").device_memset_available is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
