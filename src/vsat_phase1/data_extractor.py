from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import pytesseract


class DataExtractor:
    """
    Extracts VSAT state from webcam frames.

    - Glyphs: template matching from the assets directory.
    - Metrics: OCR via pytesseract after denoising and thresholding.
    - ActiveTurn: debounced intensity-shift signal.
    """

    def __init__(
        self,
        assets_dir: Optional[str] = None,
        metric_rois: Optional[Mapping[str, Tuple[int, int, int, int]]] = None,
        active_turn_point: Tuple[int, int] = (10, 10),
        active_turn_shift_threshold: float = 35.0,
        cooldown_seconds: float = 2.5,
        glyph_match_threshold: float = 0.78,
        tesseract_cmd: Optional[str] = None,
    ) -> None:
        module_root = Path(__file__).resolve().parent
        self.assets_dir = Path(assets_dir) if assets_dir else module_root / "assets"
        self.metric_rois: Dict[str, Tuple[int, int, int, int]] = dict(metric_rois or {})
        self.active_turn_point = active_turn_point
        self.active_turn_shift_threshold = active_turn_shift_threshold
        self.cooldown_seconds = cooldown_seconds
        self.glyph_match_threshold = glyph_match_threshold

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.ocr_config = "--oem 3 --psm 6"
        self._glyph_templates = self._load_glyph_templates(self.assets_dir)
        self._last_active_intensity: Optional[float] = None
        self._last_active_turn_timestamp = 0.0

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract glyphs, metrics, and debounced ActiveTurn signal from a frame."""
        if frame is None or frame.size == 0:
            return {
                "timestamp": time.time(),
                "active_turn": False,
                "active_turn_intensity": None,
                "glyphs": [],
                "metrics": {},
                "ocr_text": "",
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        glyphs = self._extract_glyphs(gray)
        metrics, ocr_text = self._extract_metrics(frame)
        active_turn, intensity = self._detect_active_turn(gray)

        return {
            "timestamp": time.time(),
            "active_turn": active_turn,
            "active_turn_intensity": intensity,
            "glyphs": glyphs,
            "metrics": metrics,
            "ocr_text": ocr_text,
        }

    def _load_glyph_templates(self, assets_dir: Path) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        if not assets_dir.exists():
            return templates

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for path in sorted(assets_dir.rglob("*")):
            if path.suffix.lower() not in image_extensions:
                continue
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None or template.size == 0:
                continue
            templates[path.stem] = template
        return templates

    def _extract_glyphs(self, gray_frame: np.ndarray) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if not self._glyph_templates:
            return detections

        frame_h, frame_w = gray_frame.shape[:2]
        for glyph_name, template in self._glyph_templates.items():
            th, tw = template.shape[:2]
            if th > frame_h or tw > frame_w:
                continue

            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < self.glyph_match_threshold:
                continue

            x, y = max_loc
            detections.append(
                {
                    "name": glyph_name,
                    "confidence": float(max_val),
                    "location": [int(x), int(y)],
                    "size": [int(tw), int(th)],
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def _extract_metrics(self, frame: np.ndarray) -> Tuple[Dict[str, float], str]:
        preprocessed = self._preprocess_for_ocr(frame)
        metrics: Dict[str, float] = {}
        collected_text: List[str] = []

        try:
            if self.metric_rois:
                for metric_name, roi in self.metric_rois.items():
                    crop = self._crop_roi(preprocessed, roi)
                    if crop.size == 0:
                        continue
                    text = pytesseract.image_to_string(crop, config=self.ocr_config).strip()
                    if not text:
                        continue
                    collected_text.append(f"{metric_name}: {text}")
                    value = self._extract_first_float(text)
                    if value is not None:
                        metrics[metric_name] = value
            else:
                text = pytesseract.image_to_string(preprocessed, config=self.ocr_config).strip()
                collected_text.append(text)
                metrics = self._parse_metrics_from_text(text)
        except pytesseract.TesseractNotFoundError:
            return {}, ""

        return metrics, "\n".join(t for t in collected_text if t)

    def _preprocess_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresholded = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        return thresholded

    def _crop_roi(self, image: np.ndarray, roi: Iterable[int]) -> np.ndarray:
        x, y, w, h = [int(v) for v in roi]
        ih, iw = image.shape[:2]
        x = max(0, min(iw - 1, x))
        y = max(0, min(ih - 1, y))
        w = max(1, min(iw - x, w))
        h = max(1, min(ih - y, h))
        return image[y : y + h, x : x + w]

    def _parse_metrics_from_text(self, text: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not text:
            return metrics

        def _to_num(raw: str) -> Optional[float]:
            cleaned = raw.replace(",", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None

        key_value_patterns = [
            re.compile(r"([A-Za-z][A-Za-z0-9 _-]{1,40})\s*[:=]\s*(-?\d+(?:\.\d+)?)"),
            re.compile(r"([A-Za-z][A-Za-z0-9 _-]{1,40})\s+(-?\d+(?:\.\d+)?)"),
        ]

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for pattern in key_value_patterns:
                match = pattern.search(line)
                if not match:
                    continue
                raw_key = match.group(1).strip()
                if re.search(r"\d", raw_key):
                    continue
                if re.search(r"(?:st|nd|rd|th)\s*(?:of|/)", raw_key, flags=re.IGNORECASE):
                    continue

                key = self._canonical_metric_key(raw_key)
                try:
                    metrics[key] = float(match.group(2))
                except ValueError:
                    pass
                break

        # Tournament placement pattern: "153rd of 250" or "153rd/250".
        placement_match = re.search(
            r"\b(\d{1,4})(?:st|nd|rd|th)\s*(?:of|/)\s*(\d{1,5})\b",
            text,
            flags=re.IGNORECASE,
        )
        if placement_match:
            rank = _to_num(placement_match.group(1))
            total_players = _to_num(placement_match.group(2))
            if rank is not None:
                metrics["TournamentRank"] = rank
            if total_players is not None:
                metrics["Players_Left"] = total_players
                metrics["PlayersLeft"] = total_players

        # Payout line pattern: "71st:$1.25" (position where payouts start / next pay jump).
        payout_match = re.search(
            r"\b(\d{1,4})(?:st|nd|rd|th)\s*[:\-]\s*\$?\s*([\d,]+(?:\.\d+)?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if payout_match:
            payout_position = _to_num(payout_match.group(1))
            payout_amount = _to_num(payout_match.group(2))
            if payout_position is not None:
                metrics["Payout_Proximity"] = payout_position
                metrics["PayoutProximity"] = payout_position
            if payout_amount is not None:
                metrics["NextPayout"] = payout_amount

        # Average stack depth pattern: "Avg: 69 BB".
        avg_bb_match = re.search(
            r"\bavg(?:erage)?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*bb\b",
            text,
            flags=re.IGNORECASE,
        )
        if avg_bb_match:
            avg_bb = _to_num(avg_bb_match.group(1))
            if avg_bb is not None:
                metrics["Avg_BB"] = avg_bb
                metrics["AvgBB"] = avg_bb

        # Incoming decision stake pattern: "Call 1 BB", "To Call: 2.5", "Raise 10".
        current_stake_match = re.search(
            r"\b(?:to\s*call|call|bet|raise|all[- ]?in)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:bb|b|chips?)?\b",
            text,
            flags=re.IGNORECASE,
        )
        if current_stake_match:
            current_stake = _to_num(current_stake_match.group(1))
            if current_stake is not None:
                metrics["CurrentStake"] = current_stake

        # Fallback: recover TotalValue if OCR text includes a recognizable phrase.
        if "TotalValue" not in metrics:
            total_match = re.search(
                r"total\s*value[^0-9-]*(-?\d+(?:\.\d+)?)",
                text,
                flags=re.IGNORECASE,
            )
            if total_match:
                metrics["TotalValue"] = float(total_match.group(1))

        return metrics

    def _canonical_metric_key(self, key: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", key)
        if not tokens:
            return key.strip()
        return "".join(token[:1].upper() + token[1:] for token in tokens)

    def _extract_first_float(self, text: str) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _detect_active_turn(self, gray_frame: np.ndarray) -> Tuple[bool, float]:
        frame_h, frame_w = gray_frame.shape[:2]
        px = int(np.clip(self.active_turn_point[0], 0, frame_w - 1))
        py = int(np.clip(self.active_turn_point[1], 0, frame_h - 1))
        intensity = float(gray_frame[py, px])

        active_turn = False
        if self._last_active_intensity is not None:
            shift = abs(intensity - self._last_active_intensity)
            if shift >= self.active_turn_shift_threshold:
                now = time.monotonic()
                if now - self._last_active_turn_timestamp >= self.cooldown_seconds:
                    active_turn = True
                    self._last_active_turn_timestamp = now

        self._last_active_intensity = intensity
        return active_turn, intensity
