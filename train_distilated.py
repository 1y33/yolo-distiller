from ultralytics import YOLO
import torch

teacher_model = YOLO("runs/detect/Test3/weights/best.pt")
student_model = YOLO("yolov5n1024.yaml").load("yolov5nu.pt")

Original training code (commented out)
student_model.train(
    data="visdrone.yaml",
    teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
    distillation_loss="cwd",
    epochs=100,
    batch=16,
    workers=0,
    imgsz=1024,
    exist_ok=True,
    name="distil_x_in_n"
)