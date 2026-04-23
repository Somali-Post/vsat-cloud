from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:
    from .data_extractor import DataExtractor
    from .dss_hud import DecisionHumanizer, calculate_raw_decision
except ImportError:  # pragma: no cover
    from data_extractor import DataExtractor
    from dss_hud import DecisionHumanizer, calculate_raw_decision


app = FastAPI(title="VSAT API")

# Configure CORS for browser-based frontend access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = Path(__file__).resolve().parents[2]
index_path = project_root / "static" / "index.html"
glyphs_path = project_root / "assets" / "glyphs"

extractor = DataExtractor(assets_dir=str(glyphs_path))
humanizer = DecisionHumanizer()


@app.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    return FileResponse(index_path)


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), game_type: str = Form("CASH")) -> dict[str, Any]:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    state = extractor.process_frame(frame)
    decision = calculate_raw_decision(state, game_type=game_type)

    final_action = str(decision.get("action", "Check"))
    normalized_size = humanizer.normalize_value(
        float(decision.get("raw_value", 0.0)),
        is_postflop=bool(decision.get("is_postflop", False)),
    )

    return {"action": final_action, "size": normalized_size, "state": state}
