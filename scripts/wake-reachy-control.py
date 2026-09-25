from __future__ import annotations

from reachy_mini.io.protocol import GotoTaskRequest, SetTorqueCmd
from reachy_mini.io.ws_client import WSClient
from reachy_mini.reachy_mini import (
    INIT_ANTENNAS_JOINT_POSITIONS,
    INIT_HEAD_POSE,
)
from reachy_mini.utils.interpolation import InterpolationTechnique


client = WSClient("localhost", 8000)
try:
    client.wait_for_connection(timeout=5.0)
    client.send_command(SetTorqueCmd(on=True))
    request = GotoTaskRequest(
        head=INIT_HEAD_POSE.flatten().tolist(),
        antennas=list(INIT_ANTENNAS_JOINT_POSITIONS),
        duration=2.0,
        method=InterpolationTechnique.MIN_JERK,
        body_yaw=0.0,
    )
    task_id = client.send_task_request(request)
    client.wait_for_task_completion(task_id, timeout=4.0)
finally:
    client.disconnect()
