"""
Run the automotive detector on an image, a folder of images, a video, or a webcam.

Two models work together:
  1. Road-user model  -> cars, vans, trucks, pedestrians, cyclists, trams
     (your fine-tuned KITTI model in weights/best.pt, or the pretrained COCO model)
  2. Traffic-control model -> traffic lights and stop signs (pretrained COCO model)

Every detection is drawn on the output and logged to a CSV, so you can analyse
the results in Excel or Power BI afterwards.

Examples:
    python src/detect.py --source samples/street.jpg
    python src/detect.py --source samples/drive.mp4
    python src/detect.py --source 0            # webcam
"""

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# COCO class ids that matter on the road
COCO_ROAD_USERS = {0: "pedestrian", 1: "cyclist", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COCO_TRAFFIC_CONTROL = {9: "traffic light", 11: "stop sign"}

# One fixed colour (BGR) per group so the output is easy to read
COLOURS = {
    "vehicle": (255, 140, 0),      # blue: cars, vans, trucks, buses, trams
    "vulnerable": (0, 0, 255),     # red: pedestrians / cyclists
    "traffic": (0, 215, 255),      # yellow: lights / signs
}
VULNERABLE = {"pedestrian", "person_sitting", "cyclist", "motorcycle"}


def colour_for(label: str) -> tuple:
    if label in COCO_TRAFFIC_CONTROL.values():
        return COLOURS["traffic"]
    if label in VULNERABLE:
        return COLOURS["vulnerable"]
    return COLOURS["vehicle"]


def default_road_model() -> str:
    trained = Path("weights/best.pt")
    return str(trained) if trained.exists() else "yolo26n.pt"


def is_coco_model(model: YOLO) -> bool:
    return len(model.names) == 80


def detect_frame(frame, road_model, traffic_model, conf):
    """Return a list of (label, confidence, x1, y1, x2, y2) for one frame."""
    detections = []

    if is_coco_model(road_model):
        # Pretrained model: keep only road users + traffic control from one pass
        wanted = {**COCO_ROAD_USERS, **COCO_TRAFFIC_CONTROL}
        result = road_model.predict(frame, conf=conf, classes=list(wanted), verbose=False)[0]
        for box in result.boxes:
            cls = int(box.cls)
            detections.append((wanted[cls], float(box.conf), *map(int, box.xyxy[0].tolist())))
        return detections

    # Fine-tuned model (e.g. KITTI classes) for road users
    result = road_model.predict(frame, conf=conf, verbose=False)[0]
    for box in result.boxes:
        label = road_model.names[int(box.cls)]
        if label == "misc":
            continue
        detections.append((label, float(box.conf), *map(int, box.xyxy[0].tolist())))

    # Pretrained COCO model for traffic lights / stop signs
    if traffic_model is not None:
        result = traffic_model.predict(frame, conf=conf, classes=list(COCO_TRAFFIC_CONTROL), verbose=False)[0]
        for box in result.boxes:
            label = COCO_TRAFFIC_CONTROL[int(box.cls)]
            detections.append((label, float(box.conf), *map(int, box.xyxy[0].tolist())))
    return detections


def draw(frame, detections):
    for label, conf, x1, y1, x2, y2 in detections:
        colour = colour_for(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Put the label above the box, or inside it if the box touches the top edge
        top = y1 - th - 6 if y1 - th - 6 >= 0 else y1
        cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + 6), colour, -1)
        cv2.putText(frame, text, (x1 + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    road_model = YOLO(args.model)
    traffic_model = None if is_coco_model(road_model) else YOLO(args.traffic_model)
    print(f"Road-user model: {args.model} ({len(road_model.names)} classes)")
    if traffic_model:
        print(f"Traffic-control model: {args.traffic_model}")

    csv_path = out_dir / "detections.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "frame", "class", "confidence", "x1", "y1", "x2", "y2"])

        source = args.source
        src_path = Path(source)

        # Images (single file or folder)
        if src_path.is_dir() or src_path.suffix.lower() in IMAGE_EXTS:
            images = sorted(p for p in src_path.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src_path.is_dir() else [src_path]
            for img_path in images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"Could not read {img_path}, skipping")
                    continue
                dets = detect_frame(frame, road_model, traffic_model, args.conf)
                for d in dets:
                    writer.writerow([img_path.name, 0, d[0], round(d[1], 3), *d[2:]])
                cv2.imwrite(str(out_dir / img_path.name), draw(frame, dets))
                print(f"{img_path.name}: {len(dets)} objects")
            print(f"\nDone. Annotated images + {csv_path} saved in {out_dir}/")
            return

        # Video file or webcam
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        name = "webcam.mp4" if source.isdigit() else f"{src_path.stem}_detected.mp4"
        writer_vid = cv2.VideoWriter(str(out_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect_frame(frame, road_model, traffic_model, args.conf)
            for d in dets:
                writer.writerow([name, frame_no, d[0], round(d[1], 3), *d[2:]])
            annotated = draw(frame, dets)
            writer_vid.write(annotated)
            if args.show:
                cv2.imshow("Automotive detector (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_no += 1
            if frame_no % 50 == 0:
                print(f"processed {frame_no} frames")

        cap.release()
        writer_vid.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {frame_no} frames -> {out_dir / name}, detections in {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Detect vehicles, pedestrians and traffic signs/lights.")
    p.add_argument("--source", required=True, help="image, folder, video file, or webcam index (0)")
    p.add_argument("--model", default=default_road_model(), help="road-user model (default: weights/best.pt if present)")
    p.add_argument("--traffic-model", default="yolo26n.pt", help="pretrained COCO model for lights/signs")
    p.add_argument("--conf", type=float, default=0.35, help="minimum confidence (0-1)")
    p.add_argument("--out", default="results/detections", help="output folder")
    p.add_argument("--show", action="store_true", help="show a live window (local machine only)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
