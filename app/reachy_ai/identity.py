from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sherpa_onnx

from .memory import MemoryStore


FACE_ROOT = Path(r"D:\ReachyAI\models\identity\face")
VOICE_MODEL = Path(
    r"D:\ReachyAI\models\identity\voice\3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)


@dataclass(frozen=True)
class Match:
    user_id: str | None
    confidence: float
    modality: str
    candidate_user_id: str | None = None


@dataclass(frozen=True)
class FaceObservation:
    embedding: np.ndarray
    quality: float
    center: tuple[int, int]
    area: float = 0.0
    center_distance: float = 0.0


@dataclass(frozen=True)
class IdentityDecision:
    user_id: str | None
    confidence: float
    private_memory_allowed: bool
    reason: str


class IdentityStabilizer:
    """Prevent one borderline voice sample from switching family members."""

    def __init__(self, *, immediate_voice_threshold: float = 0.66, confirmations: int = 2):
        self.immediate_voice_threshold = immediate_voice_threshold
        self.confirmations = max(2, confirmations)
        self.pending_user_id: str | None = None
        self.pending_count = 0

    def update(self, decision: IdentityDecision) -> IdentityDecision:
        if decision.reason != "voice-public" or not decision.user_id:
            self.pending_user_id = None
            self.pending_count = 0
            return decision
        if decision.confidence >= self.immediate_voice_threshold:
            self.pending_user_id = None
            self.pending_count = 0
            return IdentityDecision(
                decision.user_id,
                decision.confidence,
                False,
                "voice-confirmed",
            )
        if self.pending_user_id == decision.user_id:
            self.pending_count += 1
        else:
            self.pending_user_id = decision.user_id
            self.pending_count = 1
        if self.pending_count < self.confirmations:
            return IdentityDecision(None, 0.0, False, "voice-pending")
        self.pending_user_id = None
        self.pending_count = 0
        return IdentityDecision(
            decision.user_id,
            decision.confidence,
            False,
            "voice-confirmed",
        )


def select_centered_face(observations: list[FaceObservation]) -> FaceObservation:
    """Select the likely speaker after the robot has turned toward the sound."""
    if not observations:
        raise LookupError("No face detected")
    return min(observations, key=lambda item: item.center_distance)


def _match_templates(
    query: np.ndarray,
    templates: list[tuple[str, np.ndarray, float]],
    *,
    threshold: float,
    ambiguity_margin: float,
) -> tuple[str | None, float]:
    """Match by per-person consensus, not by one possibly contaminated sample."""
    ranked = _rank_templates(query, templates)
    if not ranked:
        return None, 0.0
    best_user, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    if best_score < threshold or best_score - second_score < ambiguity_margin:
        return None, max(0.0, best_score)
    return best_user, best_score


def _rank_templates(
    query: np.ndarray,
    templates: list[tuple[str, np.ndarray, float]],
) -> list[tuple[str, float]]:
    """Return consensus similarity for every enrolled person, best first."""
    scores_by_user: dict[str, list[float]] = {}
    for user_id, enrolled, _ in templates:
        vector = np.asarray(enrolled, dtype=np.float32).reshape(-1)
        score = float(np.dot(query, vector) / max(np.linalg.norm(vector), 1e-8))
        scores_by_user.setdefault(user_id, []).append(score)
    ranked: list[tuple[str, float]] = []
    for user_id, scores in scores_by_user.items():
        strongest = sorted(scores, reverse=True)[:3]
        ranked.append((user_id, float(np.median(strongest))))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


class FaceIdentifier:
    def __init__(self, store: MemoryStore):
        self.store = store
        detector_model = FACE_ROOT / "face_detection_yunet_2023mar.onnx"
        recognizer_model = FACE_ROOT / "face_recognition_sface_2021dec.onnx"
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model), "", (320, 320), 0.85, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")

    def observe_all(self, jpeg: bytes, max_faces: int = 6) -> list[FaceObservation]:
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid JPEG image")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            raise LookupError("No face detected")
        observations: list[FaceObservation] = []
        diagonal = max(float(np.hypot(width, height)), 1.0)
        ranked_faces = sorted(
            faces,
            key=lambda item: float(item[2] * item[3]),
            reverse=True,
        )[:max_faces]
        for face in ranked_faces:
            aligned = self.recognizer.alignCrop(image, face)
            vector = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-8)
            center = (int(face[0] + face[2] / 2), int(face[1] + face[3] / 2))
            center_distance = float(
                np.hypot(center[0] - width / 2, center[1] - height / 2) / diagonal
            )
            observations.append(
                FaceObservation(
                    vector,
                    float(face[-1]),
                    center,
                    area=float(face[2] * face[3]),
                    center_distance=center_distance,
                )
            )
        return observations

    def observe(self, jpeg: bytes) -> FaceObservation:
        observations = self.observe_all(jpeg)
        return max(observations, key=lambda item: item.area)

    def embedding(self, jpeg: bytes) -> tuple[np.ndarray, float]:
        observation = self.observe(jpeg)
        return observation.embedding, observation.quality

    def match_embedding(
        self, query: np.ndarray, quality: float = 1.0, threshold: float = 0.50
    ) -> Match:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        templates = self.store.biometric_templates("face")
        rankings = _rank_templates(query, templates)
        candidate_user = rankings[0][0] if rankings else None
        best_user, best_score = _match_templates(
            query,
            templates,
            threshold=threshold,
            ambiguity_margin=0.045,
        )
        if best_score < threshold:
            return Match(None, max(0.0, best_score), "face", candidate_user)
        return Match(
            best_user,
            min(1.0, best_score * quality),
            "face",
            candidate_user,
        )

    def rank_embedding(
        self, query: np.ndarray, quality: float = 1.0
    ) -> list[tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        return [
            (user_id, min(1.0, max(0.0, score * quality)))
            for user_id, score in _rank_templates(
                query, self.store.biometric_templates("face")
            )
        ]

    def match_visible_faces(
        self, observations: list[FaceObservation], threshold: float = 0.50
    ) -> list[Match]:
        """Assign different visible faces to different enrolled people globally."""
        if len(observations) <= 1:
            return [
                self.match_embedding(item.embedding, item.quality, threshold)
                for item in observations
            ]
        rankings = [self.rank_embedding(item.embedding, item.quality) for item in observations]
        edges = sorted(
            (
                (score, face_index, user_id)
                for face_index, scores in enumerate(rankings)
                for user_id, score in scores
                if score >= threshold
            ),
            reverse=True,
        )
        assigned_faces: set[int] = set()
        assigned_users: set[str] = set()
        result = [
            Match(
                None,
                scores[0][1] if scores else 0.0,
                "face",
                scores[0][0] if scores else None,
            )
            for scores in rankings
        ]
        for score, face_index, user_id in edges:
            if face_index in assigned_faces or user_id in assigned_users:
                continue
            result[face_index] = Match(user_id, score, "face", user_id)
            assigned_faces.add(face_index)
            assigned_users.add(user_id)
        return result

    def identify(self, jpeg: bytes, threshold: float = 0.50) -> Match:
        try:
            query, quality = self.embedding(jpeg)
        except (LookupError, ValueError):
            return Match(None, 0.0, "face")
        return self.match_embedding(query, quality, threshold)


class VoiceIdentifier:
    def __init__(self, store: MemoryStore, num_threads: int = 4):
        self.store = store
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(VOICE_MODEL), num_threads=num_threads, provider="cpu"
        )
        if not config.validate():
            raise RuntimeError("Invalid speaker embedding model")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embedding(self, samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        mono = np.asarray(samples, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate, np.clip(mono.reshape(-1), -1.0, 1.0))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            raise ValueError("Voice sample is too short")
        vector = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        return vector

    def identify(self, samples: np.ndarray, sample_rate: int = 16000, threshold: float = 0.45) -> Match:
        try:
            query = self.embedding(samples, sample_rate)
        except ValueError:
            return Match(None, 0.0, "voice")
        templates = self.store.biometric_templates("voice")
        rankings = _rank_templates(query, templates)
        candidate_user = rankings[0][0] if rankings else None
        best_user, best_score = _match_templates(
            query,
            templates,
            threshold=threshold,
            ambiguity_margin=0.04,
        )
        if best_score < threshold:
            return Match(None, max(0.0, best_score), "voice", candidate_user)
        return Match(best_user, min(1.0, best_score), "voice", candidate_user)


def fuse_identity(face: Match, voice: Match) -> IdentityDecision:
    if face.user_id and voice.user_id:
        if face.user_id != voice.user_id:
            return IdentityDecision(None, 0.0, False, "face/voice conflict")
        confidence = min(0.99, 0.55 * face.confidence + 0.45 * voice.confidence + 0.15)
        return IdentityDecision(face.user_id, confidence, True, "face+voice")
    single = face if face.user_id else voice
    # A single modality may identify the public speaker label, but must never
    # unlock private memory. Voice scores around 0.65-0.70 are common on the
    # robot microphones and were previously discarded by the old 0.72 gate.
    public_threshold = 0.64 if single.modality == "voice" else 0.68
    if single.user_id and single.confidence >= public_threshold:
        return IdentityDecision(
            single.user_id,
            single.confidence,
            False,
            f"{single.modality}-public",
        )
    return IdentityDecision(None, 0.0, False, "unknown")


def speaker_identity_from_voice(
    voice: Match, *, public_threshold: float = 0.50
) -> IdentityDecision:
    """Resolve who is speaking from voice only.

    A visible face is scene context, not proof that the visible person produced
    the current utterance. Camera centering and sound direction therefore do
    not participate in speaker identity.
    """
    if voice.user_id and voice.confidence >= public_threshold:
        return IdentityDecision(
            voice.user_id,
            voice.confidence,
            False,
            "voice-public",
        )
    return IdentityDecision(None, 0.0, False, "unknown")
