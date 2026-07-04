from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from imitation_learning.models.encoders.longclip_image_encoder import LongCLIPImageEncoder
from imitation_learning.models.encoders.longclip_text_encoder import LongCLIPTextEncoder
from imitation_learning.utils.qframe_query_modes import assign_qframe_rows


CAMERA_KEYS = ("third_person_camera", "robot0_wrist_camera")
CELL_W = 160
CELL_H = 160
LABEL_H = 40
MARGIN = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--instruction",
        default="Observe the cube's color, wait, then touch the cube of the same color.",
    )
    parser.add_argument(
        "--weights-path",
        default="/mnt/3fs1/data/tingwen.du/icra_method_dev/deps/Long-CLIP/longclip-L.pt",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--call-idx", type=int, default=-1)
    return parser.parse_args()


def crop_panel(image: Image.Image, col: int, camera_row: int) -> Image.Image:
    x0 = MARGIN + col * (CELL_W + MARGIN)
    y0 = MARGIN + camera_row * (CELL_H + LABEL_H + MARGIN) + LABEL_H
    border = 4
    return image.crop(
        (
            x0 + border,
            y0 + border,
            x0 + CELL_W - border,
            y0 + CELL_H - border,
        )
    )


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.reshape(image.height, image.width, 3)
    return data.permute(2, 0, 1).float().div(255.0)


def encode_camera_pair(
    encoder: LongCLIPImageEncoder,
    panels: dict[str, Image.Image],
    device: torch.device,
) -> torch.Tensor:
    tensors = torch.stack([pil_to_tensor(panels[key]) for key in CAMERA_KEYS], dim=0)
    tensors = tensors.to(device).unsqueeze(0)
    features = encoder(tensors).reshape(1, len(CAMERA_KEYS), -1)
    return features.mean(dim=1).squeeze(0)


def load_debug_events(debug_dir: Path) -> list[dict]:
    events = []
    with (debug_dir / "qframe_debug.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def select_events(events: list[dict], call_idx: int) -> list[dict]:
    if call_idx < 0:
        return events
    return [event for event in events if int(event["query_call_idx"]) == call_idx]


def render_grid(
    output_path: Path,
    event: dict,
    mode: str,
    query_panels: dict[str, Image.Image],
    history_panels: list[dict[str, Image.Image]],
    rows: list[dict[str, object]],
) -> None:
    colors = {
        "query": (220, 60, 60),
        "high": (20, 150, 50),
        "low": (40, 90, 220),
        "candidate": (120, 120, 120),
        "history": (200, 200, 200),
    }
    columns = 1 + len(rows)
    width = MARGIN + columns * (CELL_W + MARGIN)
    height = MARGIN + len(CAMERA_KEYS) * (CELL_H + LABEL_H + MARGIN)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    panels: list[tuple[str, str, float | None, dict[str, Image.Image]]] = [
        ("query", "query", None, query_panels)
    ]
    for row in rows:
        idx = int(row["history_index"])
        label = str(row.get("history_call_idx", idx))
        panels.append((label, str(row["kind"]), float(row["score"]), history_panels[idx]))

    for col, (label, kind, score, panel_images) in enumerate(panels):
        x0 = MARGIN + col * (CELL_W + MARGIN)
        for row_idx, camera_key in enumerate(CAMERA_KEYS):
            y0 = MARGIN + row_idx * (CELL_H + LABEL_H + MARGIN)
            img = panel_images[camera_key].resize((CELL_W, CELL_H))
            canvas.paste(img, (x0, y0 + LABEL_H))
            draw.rectangle(
                [x0, y0 + LABEL_H, x0 + CELL_W - 1, y0 + LABEL_H + CELL_H - 1],
                outline=colors.get(kind, (160, 160, 160)),
                width=4,
            )
            text = f"{label} {camera_key.replace('_camera', '')}"
            if score is not None and row_idx == 0:
                text = f"{label} {kind} s={score:.3f}"
            elif row_idx == 0:
                text = f"{label} {kind}"
            draw.text((x0 + 3, y0 + 3), text, fill=(0, 0, 0))
    draw.text(
        (MARGIN, height - 18),
        f"mode={mode} episode={event['episode_id']} query_call={event['query_call_idx']}",
        fill=(0, 0, 0),
    )
    canvas.save(output_path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    device = torch.device(args.device)
    image_encoder = LongCLIPImageEncoder(
        weights_path=args.weights_path,
        model_name="longclip-L",
        pretrained=True,
        frozen=True,
        feature_aggregation="map",
        apply_image_norm=True,
        input_resolution=224,
        image_meta={
            "name": "qframe_debug_crop",
            "data_type": "image",
            "shape": [3, CELL_H - 8, CELL_W - 8],
            "length": len(CAMERA_KEYS),
            "normalizer": "identity",
            "augmentation": None,
            "source_entry_names": ["qframe_debug_crop"],
        },
    ).to(device)
    image_encoder.eval()
    text_encoder = LongCLIPTextEncoder(args.weights_path).to(device)
    text_encoder.eval()
    text_query = text_encoder.encode_text([args.instruction]).to(device)

    csv_path = output_dir / "rerank_scores.csv"
    jsonl_path = output_dir / "rerank_events.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f, jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as jsonl_f:
        writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "debug_dir",
                "mode",
                "episode_id",
                "query_call_idx",
                "history_index",
                "history_call_idx",
                "kind",
                "score",
                "image_score",
                "text_score",
                "fused_score",
            ],
        )
        writer.writeheader()

        for raw_debug_dir in args.debug_dir:
            debug_dir = Path(raw_debug_dir)
            for event in select_events(load_debug_events(debug_dir), args.call_idx):
                image = Image.open(event["image"]).convert("RGB")
                query_panels = {
                    key: crop_panel(image, 0, row_idx)
                    for row_idx, key in enumerate(CAMERA_KEYS)
                }
                history_panels = []
                for hist_idx in range(len(event["rows"])):
                    col = hist_idx + 1
                    history_panels.append(
                        {
                            key: crop_panel(image, col, row_idx)
                            for row_idx, key in enumerate(CAMERA_KEYS)
                        }
                    )

                image_query = encode_camera_pair(image_encoder, query_panels, device)
                history_keys = torch.stack(
                    [
                        encode_camera_pair(image_encoder, panels, device)
                        for panels in history_panels
                    ],
                    dim=0,
                )
                image_scores = torch.einsum(
                    "d,hd->h",
                    F.normalize(image_query, dim=-1, eps=1e-6),
                    F.normalize(history_keys, dim=-1, eps=1e-6),
                )
                text_scores = torch.einsum(
                    "bd,hd->bh",
                    F.normalize(text_query, dim=-1, eps=1e-6),
                    F.normalize(history_keys, dim=-1, eps=1e-6),
                ).squeeze(0)
                fused_query = (
                    args.alpha * F.normalize(image_query, dim=-1, eps=1e-6)
                    + (1.0 - args.alpha)
                    * F.normalize(text_query.squeeze(0), dim=-1, eps=1e-6)
                )
                fused_scores = torch.einsum(
                    "d,hd->h",
                    F.normalize(fused_query, dim=-1, eps=1e-6),
                    F.normalize(history_keys, dim=-1, eps=1e-6),
                )
                mode_scores = {
                    "image_only": image_scores,
                    "text_only": text_scores,
                    "image_text_fused": fused_scores,
                }
                history_call_idx = [
                    row.get("history_call_idx", row["history_index"])
                    for row in event["rows"]
                ]

                for mode, scores in mode_scores.items():
                    rows = assign_qframe_rows(
                        scores.detach().cpu(),
                        max_candidates=int(event["max_candidates"]),
                        high_topk=int(event["high_topk"]),
                        low_topk=int(event["low_topk"]),
                    )
                    for row, call in zip(rows, history_call_idx):
                        row["history_call_idx"] = call

                    image_name = (
                        f"{debug_dir.name}_call{int(event['query_call_idx']):03d}_{mode}.png"
                    )
                    render_grid(
                        output_dir / "images" / image_name,
                        event,
                        mode,
                        query_panels,
                        history_panels,
                        rows,
                    )
                    record = {
                        "debug_dir": str(debug_dir),
                        "mode": mode,
                        "episode_id": int(event["episode_id"]),
                        "query_call_idx": int(event["query_call_idx"]),
                        "image": str(output_dir / "images" / image_name),
                        "rows": rows,
                    }
                    jsonl_f.write(json.dumps(record, sort_keys=True) + "\n")
                    for row_idx, row in enumerate(rows):
                        writer.writerow(
                            {
                                "debug_dir": str(debug_dir),
                                "mode": mode,
                                "episode_id": int(event["episode_id"]),
                                "query_call_idx": int(event["query_call_idx"]),
                                "history_index": row_idx,
                                "history_call_idx": history_call_idx[row_idx],
                                "kind": row["kind"],
                                "score": f"{float(row['score']):.6f}",
                                "image_score": f"{float(image_scores[row_idx].detach().cpu()):.6f}",
                                "text_score": f"{float(text_scores[row_idx].detach().cpu()):.6f}",
                                "fused_score": f"{float(fused_scores[row_idx].detach().cpu()):.6f}",
                            }
                        )

    print(f"wrote {csv_path}")
    print(f"wrote {jsonl_path}")


if __name__ == "__main__":
    main()
