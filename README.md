# 🚗 Automotive Object Detection with YOLO26

Real-time detection of **cars, vans, trucks, trams, pedestrians, cyclists and traffic lights** in street scenes, the core perception task behind driver-assistance (ADAS) features such as emergency braking, pedestrian warning and traffic-light recognition.

A pretrained **Ultralytics YOLO26** model is fine-tuned on **KITTI**, a benchmark of real driving footage recorded in Karlsruhe, Germany. Every detection is also logged to CSV, so results can be analysed in Excel or Power BI rather than only viewed as pictures.

![Detections on KITTI validation scenes](assets/demo_grid.jpg)
<sub>Detections on unseen KITTI validation images. Blue = vehicles, red = pedestrians & cyclists, yellow = traffic lights.</sub>

---

## Results

Evaluated on the **1,496 KITTI validation images** (8,128 labelled objects) the model never saw during training.

| Metric | After epoch 1 | Final model |
|---|---|---|
| **mAP@0.5** | 0.283 | **0.545** |
| mAP@0.5:0.95 | 0.154 | **0.309** |
| Precision | 0.510 | **0.656** |
| Recall | 0.249 | **0.491** |
| Inference time | – | **17.7 ms / image on a 2-core CPU** (no GPU) |

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| car | 0.784 | 0.775 | **0.844** | 0.556 |
| truck | 0.866 | 0.639 | 0.757 | 0.505 |
| tram | 0.629 | 0.573 | 0.613 | 0.306 |
| van | 0.624 | 0.512 | 0.561 | 0.353 |
| pedestrian | 0.622 | 0.479 | 0.506 | 0.209 |
| person_sitting | 0.476 | 0.333 | 0.350 | 0.154 |
| cyclist | 0.534 | 0.318 | 0.344 | 0.160 |

<sub>Training: YOLO26-nano, 7 epochs, 416 px images, CPU only. Full metrics in [`results/`](results/).</sub>

![Per-class accuracy](assets/metrics_per_class.png)

![Training curves](assets/training_curves.png)

### Key observations

- **Cars are detected reliably (mAP@0.5 = 0.84).** Cars make up about 70% of all labelled objects in KITTI, so the model saw the most examples of them. Accuracy follows data volume: the rarer the class, the weaker the result.
- **Pedestrians and cyclists are the weak spot (recall 0.48 and 0.32).** They are small in the image, less frequent in the data, and the training used a reduced 416 px resolution to fit CPU compute. For a safety system this matters most: a missed pedestrian is far more costly than a false alarm, so improving recall on these classes is the clear next priority.
- **Real-time speed without a GPU.** At 17.7 ms per image on two CPU cores (~55 frames per second for the model itself), the nano model is fast enough for live video, which supports the case for small, efficient models on in-vehicle hardware.
- **Accuracy was still rising when training stopped** (0.28 → 0.55 mAP@0.5 over 7 epochs). Longer training on a GPU at 640 px would raise all scores, especially for small objects.

### From detections to data

`detect.py` writes every detection to a CSV (source, frame, class, confidence, bounding box). `summarize_detections.py` turns it into a traffic report. Example from 40 validation scenes: 91 cars, 8 pedestrians, 6 traffic lights, 4 trucks, 4 vans and 2 cyclists, with an average confidence of 0.66.

![Traffic report](assets/traffic_report.png)

---

## How it works

```
 camera frame ──► YOLO26 fine-tuned on KITTI ──► cars, vans, trucks, trams,
       │                                          pedestrians, cyclists
       └────────► YOLO26 pretrained on COCO ───► traffic lights, stop signs
                                                        │
                       annotated image / video  ◄───────┤
                       detections.csv (class, confidence, box, frame) ◄┘
```

| Decision | Why |
|---|---|
| **YOLO26 nano** | Newest Ultralytics model family (January 2026), designed for edge devices with NMS-free end-to-end inference: the kind of hardware constraint in-vehicle systems have. |
| **Transfer learning** instead of training from scratch | Reuses visual features learned on the COCO dataset, so useful accuracy is reached in a few epochs with modest compute. |
| **KITTI dataset** | Real German road scenes with labelled cars, vans, trucks, pedestrians, cyclists and trams. |
| **Two-model pipeline** | KITTI has no traffic-light or sign labels, so a pretrained COCO model covers those classes without labelling new data. |
| **CSV logging** | Turns detections into analysable data: counts per class, objects in view over time, confidence distribution. |

---

## Project structure

```
automotive-object-detection/
├── configs/kitti.yaml              # dataset definition (auto-downloads KITTI, ~390 MB)
├── src/
│   ├── train.py                    # fine-tune YOLO26 on KITTI
│   ├── evaluate.py                 # per-class precision / recall / mAP → CSV, chart, markdown
│   ├── detect.py                   # image, folder, video or webcam → annotated output + CSV
│   └── summarize_detections.py     # traffic report from detections.csv
├── notebooks/train_on_colab.ipynb  # retrain on a free Google Colab GPU
├── weights/best.pt                 # trained model (5.4 MB)
├── results/                        # metrics, training log, sample detections
└── assets/                         # images used in this README
```

---

## Quick start

Download or clone this repository, then:

```bash
pip install -r requirements.txt

python src/detect.py --source path/to/dashcam.mp4    # detect with the trained model
python src/summarize_detections.py                    # traffic report from that run
python src/evaluate.py                                # reproduce the metrics above
python src/train.py --epochs 50 --device 0            # retrain (GPU recommended)
```

Or open `notebooks/train_on_colab.ipynb` in Google Colab to retrain on a free GPU.

---

## Limitations & next steps

- **Small and vulnerable road users:** train longer at 640 px and add more pedestrian and cyclist examples to raise recall.
- **Traffic signs:** only stop signs are covered by the COCO model. Next step: fine-tune on the **German Traffic Sign Detection Benchmark (GTSDB)** to recognise speed limits, give-way and other German signs.
- **Object tracking:** add ByteTrack (built into Ultralytics) to follow each vehicle across video frames and count unique road users.
- **Edge deployment:** export to ONNX or TensorRT and benchmark latency on embedded hardware.
- **Conditions:** KITTI is daytime, good-weather driving. Night, rain and snow would need additional data such as BDD100K.

---

## Tech stack

Python · Ultralytics YOLO26 · PyTorch · OpenCV · pandas · matplotlib · Google Colab

## Credits & licences

- Model and training framework: [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- Dataset and demo images: [KITTI Vision Benchmark](https://www.cvlibs.net/datasets/kitti/), Karlsruhe Institute of Technology & Toyota Technological Institute at Chicago (CC BY-NC-SA 3.0, non-commercial use)
- Project idea from [project-based-learning](https://github.com/practical-tutorials/project-based-learning)

This project is released under the AGPL-3.0 licence to match Ultralytics.

**Author:** Souad Essemyhij, MSc Robotics and Autonomous Systems, Universität zu Lübeck · [LinkedIn](https://www.linkedin.com/in/souad-essemyhi-742a9520b/) · [GitHub](https://github.com/souadessemyhij1999)
