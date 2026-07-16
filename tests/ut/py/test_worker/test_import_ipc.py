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
    _CTRL_IMPORT_IPC,
    _IPC_REPLY_HEADER,
    _IPC_REPLY_RECORD,
    _decode_ipc_import_payload,
    _encode_ipc_import_payload,
)


def _initialized_worker(device_ids=(8, 9)):
    worker = Worker.__new__(Worker)
    worker._worker = MagicMock()
    worker._initialized = True
    worker._config = {"device_ids": list(device_ids)}
    worker._py_control_timeout_s = 30.0
    return worker


def test_ipc_payload_round_trip():
    keys = {9: b"b" * 256, 8: b"a" * 256}
    payload = _encode_ipc_import_payload("reply-name", keys)
    reply_name, decoded = _decode_ipc_import_payload(payload)
    assert reply_name == "reply-name"
    assert decoded == {8: b"a" * 256, 9: b"b" * 256}


@pytest.mark.parametrize("size", [0, 255, 257])
def test_ipc_payload_rejects_non_256_byte_key(size):
    with pytest.raises(ValueError, match="exactly 256 bytes"):
        _encode_ipc_import_payload("reply-name", {8: b"x" * size})


def test_import_ipc_all_broadcasts_and_reads_child_replies():
    worker = _initialized_worker()

    def broadcast(worker_type, sub_cmd, payload, digest, *, timeout_s):
        assert worker_type == WorkerType.NEXT_LEVEL
        assert sub_cmd == _CTRL_IMPORT_IPC
        assert digest is None
        assert timeout_s == 30.0
        reply_name, keys = _decode_ipc_import_payload(payload)
        assert keys == {8: b"a" * 256, 9: b"b" * 256}

        from multiprocessing.shared_memory import SharedMemory

        reply = SharedMemory(name=reply_name)
        try:
            count = _IPC_REPLY_HEADER.unpack_from(reply.buf, 0)[0]
            assert count == 2
            for index, device_id in enumerate(sorted(keys)):
                offset = _IPC_REPLY_HEADER.size + index * _IPC_REPLY_RECORD.size
                _IPC_REPLY_RECORD.pack_into(
                    reply.buf, offset, device_id, 0x10000000 + device_id
                )
        finally:
            reply.close()
        return [SimpleNamespace(ok=True, worker_type="NEXT_LEVEL", worker_id=0, error_message="")]

    worker._worker.broadcast_control_all.side_effect = broadcast
    assert worker.import_ipc_all({8: b"a" * 256, 9: b"b" * 256}) == {
        8: 0x10000008,
        9: 0x10000009,
    }


def test_import_ipc_all_requires_exact_worker_device_set():
    worker = _initialized_worker()
    with pytest.raises(ValueError, match="exactly match Worker device_ids"):
        worker.import_ipc_all({8: b"a" * 256})
    worker._worker.broadcast_control_all.assert_not_called()


def test_import_ipc_all_requires_initialized_worker():
    worker = _initialized_worker()
    worker._initialized = False
    with pytest.raises(RuntimeError, match=r"Worker\.init"):
        worker.import_ipc_all({8: b"a" * 256, 9: b"b" * 256})
