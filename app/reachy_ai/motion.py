from __future__ import annotations

import math
import re
import time
from typing import Any, Literal

from reachy_mini.motion.recorded_move import RecordedMoves


ActionName = Literal[
    "nod",
    "shake",
    "look_left",
    "look_right",
    "look_up",
    "look_down",
    "center",
    "body_left",
    "body_right",
    "body_center",
    "body_wiggle",
    "body_wiggle_large",
    "antenna_wiggle",
    "antenna_center",
]

ALLOWED_ACTIONS = {
    "nod",
    "shake",
    "look_left",
    "look_right",
    "look_up",
    "look_down",
    "center",
    "body_left",
    "body_right",
    "body_center",
    "body_wiggle",
    "body_wiggle_large",
    "antenna_wiggle",
    "antenna_center",
}

ACTION_TAG = re.compile(
    r"\[\[ACTION:(nod|shake|look_left|look_right|look_up|look_down|center|"
    r"body_left|body_right|body_center|body_wiggle|body_wiggle_large|"
    r"antenna_wiggle|antenna_center)\]\]",
    re.IGNORECASE,
)
ANY_ACTION_TAG = re.compile(r"\[\[ACTION:[^\]\n]{1,64}\]\]", re.IGNORECASE)

OFFICIAL_EMOTION_NAMES = frozenset(
    "amazed1 anxiety1 attentive1 attentive2 boredom1 boredom2 calming1 cheerful1 "
    "come1 confused1 contempt1 curious1 dance1 dance2 dance3 disgusted1 displeased1 "
    "displeased2 downcast1 dying1 electric1 enthusiastic1 enthusiastic2 exhausted1 "
    "fear1 frustrated1 furious1 go_away1 grateful1 helpful1 helpful2 impatient1 "
    "impatient2 incomprehensible2 indifferent1 inquiring1 inquiring2 inquiring3 "
    "irritated1 irritated2 laughing1 laughing2 lonely1 lost1 loving1 mini-deep-sleep "
    "no1 no_excited1 no_sad1 oops1 oops2 proud1 proud2 proud3 rage1 relief1 relief2 "
    "reprimand1 reprimand2 reprimand3 resigned1 sad1 sad2 scared1 serenity1 shy1 "
    "sleep1 success1 success2 surprised1 surprised2 thoughtful1 thoughtful2 tired1 "
    "toc-toc-toc uncertain1 uncomfortable1 understanding1 understanding2 waiting "
    "wake-mini-up welcoming1 welcoming2 yes1 yes_sad1".split()
)
EMOTION_TAG = re.compile(r"\[\[EMOTION:([a-z0-9_-]+)\]\]", re.IGNORECASE)

_EMOTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("laughing2", re.compile(r"(?:开心|高兴|快乐|大笑|笑一个).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("welcoming2", re.compile(r"(?:欢迎|打招呼|问候).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("thoughtful1", re.compile(r"(?:思考|想一想).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("confused1", re.compile(r"(?:困惑|疑惑|不明白).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("sad1", re.compile(r"(?:伤心|难过|悲伤).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("rage1", re.compile(r"(?:生气|愤怒|发火).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("surprised1", re.compile(r"(?:惊讶|吃惊).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("scared1", re.compile(r"(?:害怕|恐惧).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("grateful1", re.compile(r"(?:感激|感谢).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("loving1", re.compile(r"(?:喜欢|爱心|喜爱).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("success1", re.compile(r"(?:成功|庆祝|胜利).{0,5}(?:表情|动作)?", re.IGNORECASE)),
    ("dance3", re.compile(r"(?:跳舞|舞蹈|跳一段)", re.IGNORECASE)),
    ("exhausted1", re.compile(r"(?:累了|疲惫|筋疲力尽).{0,5}(?:表情|动作)?", re.IGNORECASE)),
)

_emotion_library: RecordedMoves | None = None

_REQUEST_PATTERNS: tuple[tuple[ActionName, re.Pattern[str]], ...] = (
    ("nod", re.compile(r"点(?:个|一下|两下)?头|点点头|nod", re.IGNORECASE)),
    ("shake", re.compile(r"摇(?:个|一下|两下)?头|摇摇头|shake your head", re.IGNORECASE)),
    (
        "body_left",
        re.compile(
            r"(?:身体|身子|底座).{0,5}(?:向|往)?左(?:边)?(?:转|转动|旋转)|"
            r"(?:向|往)?左(?:边)?(?:转|转动)(?:一下)?(?:身体|身子|底座)|turn (?:your )?body left",
            re.IGNORECASE,
        ),
    ),
    (
        "body_right",
        re.compile(
            r"(?:身体|身子|底座).{0,5}(?:向|往)?右(?:边)?(?:转|转动|旋转)|"
            r"(?:向|往)?右(?:边)?(?:转|转动)(?:一下)?(?:身体|身子|底座)|turn (?:your )?body right",
            re.IGNORECASE,
        ),
    ),
    (
        "body_center",
        re.compile(
            r"(?:身体|身子|底座).{0,6}(?:回正|转回来|复位)|center (?:your )?body",
            re.IGNORECASE,
        ),
    ),
    (
        "body_wiggle_large",
        re.compile(
            r"(?:身体|身子|底座).{0,8}(?:大幅|大一点|幅度大).{0,6}(?:动|转|转动|旋转)|"
            r"(?:大幅|大一点|幅度大).{0,6}(?:动|转|转动|旋转).{0,8}(?:身体|身子|底座)",
            re.IGNORECASE,
        ),
    ),
    (
        "body_wiggle",
        re.compile(
            r"(?:动|转|转动|晃)(?:一下|一动|一转)?(?:你的)?(?:身体|身子|底座)|"
            r"(?:身体|身子|底座).{0,5}(?:动一下|动一动|转一下|转一转|转动一下|晃一下)|"
            r"move (?:your )?body|turn (?:your )?body$",
            re.IGNORECASE,
        ),
    ),
    (
        "antenna_wiggle",
        re.compile(
            r"(?:动动|摇摇|晃晃|摆摆|转动)(?:一下|一动|一转)?(?:你的)?(?:天线|触角)|"
            r"(?:动|摇|晃|摆|转|转动)(?:一下|一动|一转)?(?:你的)?(?:天线|触角)|"
            r"(?:天线|触角).{0,6}(?:动一下|动一动|摇一下|摇一摇|晃一下|晃一晃|摆一摆|转一下|转动一下)|"
            r"wiggle (?:your )?antennas?",
            re.IGNORECASE,
        ),
    ),
    (
        "antenna_center",
        re.compile(
            r"(?:天线|触角).{0,5}(?:回正|复位|恢复)|center (?:your )?antennas?",
            re.IGNORECASE,
        ),
    ),
    (
        "look_left",
        re.compile(
            r"(?:向|往)?左(?:边)?看|(?:头|脑袋).{0,4}(?:向|往)?左(?:边)?(?:转|扭)|look left",
            re.IGNORECASE,
        ),
    ),
    (
        "look_right",
        re.compile(
            r"(?:向|往)?右(?:边)?看|(?:头|脑袋).{0,4}(?:向|往)?右(?:边)?(?:转|扭)|look right",
            re.IGNORECASE,
        ),
    ),
    ("look_up", re.compile(r"抬头|(?:向|往)?上看|look up", re.IGNORECASE)),
    ("look_down", re.compile(r"低头|(?:向|往)?下看|look down", re.IGNORECASE)),
    (
        "center",
        re.compile(r"回正|看(?:向)?前面|看(?:向)?正前方|转回来|look forward|center", re.IGNORECASE),
    ),
)

_NEGATION = re.compile(r"(?:别|不要|不用|不必|停止|取消)\s*$", re.IGNORECASE)


def requested_actions(text: str) -> list[ActionName]:
    """Extract explicit, safe movement requests without involving an LLM."""
    found: list[tuple[int, ActionName]] = []
    for action, pattern in _REQUEST_PATTERNS:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 5) : match.start()]
            if _NEGATION.search(prefix):
                continue
            found.append((match.start(), action))
    found.sort(key=lambda item: item[0])
    result: list[ActionName] = []
    for _, action in found:
        if action not in result:
            result.append(action)
    # Prefer the most specific body command. A directional or explicitly large
    # request can also match the generic body-wiggle wording; do not execute both.
    if "body_wiggle_large" in result and "body_wiggle" in result:
        result.remove("body_wiggle")
    if ("body_left" in result or "body_right" in result) and "body_wiggle" in result:
        result.remove("body_wiggle")
    return result[:3]


def requested_emotions(text: str) -> list[str]:
    """Resolve explicit natural-language or exact official emotion requests."""
    normalized = text.casefold()
    found: list[tuple[int, str]] = []
    for name in OFFICIAL_EMOTION_NAMES:
        position = normalized.find(name)
        if position >= 0:
            found.append((position, name))
    for name, pattern in _EMOTION_PATTERNS:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 5) : match.start()]
            if not _NEGATION.search(prefix):
                found.append((match.start(), name))
    found.sort(key=lambda item: item[0])
    result: list[str] = []
    for _, name in found:
        if name not in result:
            result.append(name)
    return result[:2]


def strip_action_tags(text: str) -> tuple[str, list[ActionName]]:
    actions: list[ActionName] = []
    for match in ACTION_TAG.finditer(text):
        action = match.group(1).lower()
        if action in ALLOWED_ACTIONS and action not in actions:
            actions.append(action)  # type: ignore[arg-type]
    # Unknown model-generated action names must never be spoken aloud or
    # executed.  The allow-listed pattern above extracts executable actions;
    # this broader pattern only removes malformed/unsupported control markup.
    cleaned = ANY_ACTION_TAG.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, actions[:3]


def strip_emotion_tags(text: str) -> tuple[str, list[str]]:
    emotions: list[str] = []
    for match in EMOTION_TAG.finditer(text):
        name = match.group(1).lower()
        if name in OFFICIAL_EMOTION_NAMES and name not in emotions:
            emotions.append(name)
    cleaned = EMOTION_TAG.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, emotions[:2]


def merge_actions(*groups: list[ActionName]) -> list[ActionName]:
    result: list[ActionName] = []
    for group in groups:
        for action in group:
            if action in ALLOWED_ACTIONS and action not in result:
                result.append(action)
    return result[:3]


BODY_TURN_RADIANS = math.radians(60.0)
BODY_WIGGLE_RADIANS = math.radians(50.0)
BODY_WIGGLE_LARGE_RADIANS = math.radians(90.0)
SOUND_BODY_ASSIST_START_RADIANS = math.radians(15.0)
SOUND_BODY_MAX_RADIANS = math.radians(60.0)
SOUND_BODY_SHARE = 0.65
SOUND_WORLD_GAZE_MAX_RADIANS = math.radians(85.0)
SOUND_BODY_ABSOLUTE_LIMIT = math.radians(150.0)
COORDINATED_BODY_TURN_RADIANS = math.radians(60.0)
COORDINATED_WORLD_GAZE_RADIANS = math.radians(78.0)
ANTENNA_NEUTRAL = (-0.1745, 0.1745)
ANTENNA_WIGGLE_RADIANS = math.radians(20.0)


def _goto_gaze(
    reachy: Any, x: float, y: float, z: float, *, duration: float, body_yaw: float | None = None
) -> None:
    """Move the head without implicitly resetting the current body yaw."""
    head = reachy.look_at_world(x, y, z, duration=duration, perform_movement=False)
    reachy.goto_target(head=head, duration=duration, body_yaw=body_yaw)


def _goto_body(reachy: Any, yaw: float, *, duration: float = 0.65) -> None:
    reachy.goto_target(head=None, antennas=None, duration=duration, body_yaw=float(yaw))


def _present_body_yaw(reachy: Any) -> float:
    """Read the actual body yaw from the SDK live joint state."""
    try:
        head_joints, _ = reachy.get_current_joint_positions()
        if head_joints:
            return float(head_joints[0])
    except Exception:
        pass
    return 0.0


def _goto_world_yaw(reachy: Any, yaw: float, *, duration: float = 0.50) -> None:
    """Aim the head at a world-frame yaw while preserving current body yaw."""
    yaw = math.atan2(math.sin(float(yaw)), math.cos(float(yaw)))
    _goto_gaze(
        reachy,
        math.cos(yaw),
        math.sin(yaw),
        0.0,
        duration=duration,
        body_yaw=None,
    )


def _coordinated_side_turn(reachy: Any, direction: int) -> None:
    """Turn body first, then aim head farther in the same world direction."""
    sign = 1.0 if direction > 0 else -1.0
    _goto_body(reachy, sign * COORDINATED_BODY_TURN_RADIANS, duration=0.78)
    _goto_world_yaw(
        reachy,
        sign * COORDINATED_WORLD_GAZE_RADIANS,
        duration=0.48,
    )


def _goto_antennas(
    reachy: Any, antennas: tuple[float, float], *, duration: float = 0.25
) -> None:
    reachy.goto_target(
        head=None,
        antennas=[float(antennas[0]), float(antennas[1])],
        duration=duration,
        body_yaw=None,
    )


def perform_actions(reachy: Any, actions: list[str]) -> None:
    """Execute fixed, conservative head, body-yaw and antenna gestures."""
    pending = list(actions[:3])
    if "look_left" in pending and "body_left" in pending:
        _coordinated_side_turn(reachy, 1)
        pending = [
            action for action in pending if action not in {"look_left", "body_left"}
        ]
    elif "look_right" in pending and "body_right" in pending:
        _coordinated_side_turn(reachy, -1)
        pending = [
            action for action in pending if action not in {"look_right", "body_right"}
        ]

    for action in pending:
        if action == "nod":
            _goto_gaze(reachy, 1.0, 0.0, -0.22, duration=0.30)
            _goto_gaze(reachy, 1.0, 0.0, 0.12, duration=0.30)
            _goto_gaze(reachy, 1.0, 0.0, 0.0, duration=0.28)
        elif action == "shake":
            _goto_gaze(reachy, 1.0, 0.30, 0.0, duration=0.30)
            _goto_gaze(reachy, 1.0, -0.30, 0.0, duration=0.42)
            _goto_gaze(reachy, 1.0, 0.0, 0.0, duration=0.30)
        elif action == "look_left":
            _goto_gaze(reachy, 1.0, 0.55, 0.0, duration=0.55)
        elif action == "look_right":
            _goto_gaze(reachy, 1.0, -0.55, 0.0, duration=0.55)
        elif action == "look_up":
            _goto_gaze(reachy, 1.0, 0.0, 0.30, duration=0.50)
        elif action == "look_down":
            _goto_gaze(reachy, 1.0, 0.0, -0.28, duration=0.50)
        elif action == "center":
            _goto_gaze(reachy, 1.0, 0.0, 0.0, duration=0.50)
        elif action == "body_left":
            _goto_body(reachy, BODY_TURN_RADIANS)
        elif action == "body_right":
            _goto_body(reachy, -BODY_TURN_RADIANS)
        elif action == "body_center":
            _goto_body(reachy, 0.0)
        elif action == "body_wiggle":
            _goto_body(reachy, BODY_WIGGLE_RADIANS, duration=0.75)
            _goto_body(reachy, -BODY_WIGGLE_RADIANS, duration=1.0)
            _goto_body(reachy, 0.0, duration=0.75)
        elif action == "body_wiggle_large":
            _goto_body(reachy, BODY_WIGGLE_LARGE_RADIANS, duration=1.1)
            _goto_body(reachy, -BODY_WIGGLE_LARGE_RADIANS, duration=1.5)
            _goto_body(reachy, 0.0, duration=1.1)
        elif action == "antenna_wiggle":
            right, left = ANTENNA_NEUTRAL
            swing = ANTENNA_WIGGLE_RADIANS
            _goto_antennas(reachy, (right - swing, left + swing), duration=0.22)
            _goto_antennas(reachy, (right + swing, left - swing), duration=0.22)
            _goto_antennas(reachy, ANTENNA_NEUTRAL, duration=0.28)
        elif action == "antenna_center":
            _goto_antennas(reachy, ANTENNA_NEUTRAL, duration=0.30)


def sound_turn_components(angle_radians: float) -> tuple[float, float | None]:
    """Return world-frame gaze yaw and body yaw for a sound direction.

    ReSpeaker's useful frontal half-plane is represented here as 0..pi, with
    pi/2 straight ahead, 0 to the left and pi to the right.  For meaningful
    side angles the body takes most of the turn so Reachy faces the speaker
    instead of leaving the base fixed while only twisting its neck.

    Important: the head target returned here is a WORLD-FRAME yaw, not the
    residual angle relative to the body. Reachy's look_at_world() works in the
    neutral/world frame, and the SDK accounts for body yaw when solving the
    head joints.
    """
    angle = max(0.0, min(math.pi, float(angle_radians)))
    signed_offset = (math.pi / 2.0) - angle
    body_yaw: float | None = None

    if abs(signed_offset) >= SOUND_BODY_ASSIST_START_RADIANS:
        body_yaw = max(
            -SOUND_BODY_MAX_RADIANS,
            min(SOUND_BODY_MAX_RADIANS, signed_offset * SOUND_BODY_SHARE),
        )

    return signed_offset, body_yaw


def look_toward_sound(reachy: Any, angle_radians: float) -> None:
    """Turn body first, then aim the head at the same world-frame sound source."""
    local_offset, body_delta = sound_turn_components(angle_radians)
    current_body_yaw = _present_body_yaw(reachy)
    world_gaze_yaw = current_body_yaw + local_offset
    body_yaw = (
        None
        if body_delta is None
        else max(
            -SOUND_BODY_ABSOLUTE_LIMIT,
            min(SOUND_BODY_ABSOLUTE_LIMIT, current_body_yaw + body_delta),
        )
    )
    body_fraction = (
        0.0
        if body_yaw is None
        else min(1.0, abs(body_yaw) / SOUND_BODY_MAX_RADIANS)
    )
    if body_yaw is not None:
        _goto_body(
            reachy,
            body_yaw,
            duration=0.50 + 0.35 * body_fraction,
        )
    _goto_world_yaw(
        reachy,
        world_gaze_yaw,
        duration=0.38 + 0.18 * (1.0 - body_fraction),
    )


def perform_emotions(reachy: Any, emotions: list[str]) -> None:
    """Play official recorded emotion trajectories through the SDK target API."""
    global _emotion_library
    if _emotion_library is None:
        _emotion_library = RecordedMoves(
            "pollen-robotics/reachy-mini-emotions-library"
        )
    for name in emotions[:2]:
        if name not in OFFICIAL_EMOTION_NAMES:
            continue
        move = _emotion_library.get(name)
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= move.duration:
                break
            head, antennas, body_yaw = move.evaluate(elapsed)
            reachy.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            time.sleep(1.0 / 60.0)
        reachy.look_at_world(1.0, 0.0, 0.0, duration=0.45)
