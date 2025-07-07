from ultralytics import YOLO
import torch

teacher_model = YOLO("runs/detect/Test3/weights/best.pt")
student_model = YOLO("yolov5n1024.yaml").load("yolov5nu.pt")
test_model = YOLO("yolov5n.yaml")

# Create a dummy input tensor (batch_size=1, channels=3, height=1024, width=1024)
dummy_input = torch.randn(1, 3, 1024, 1024)

# Set models to evaluation mode
teacher_model.model.eval()
student_model.model.eval()
test_model.model.eval()

# Forward pass for teacher model
with torch.no_grad():
    teacher_output = teacher_model.model(dummy_input)
    print(f"Teacher model output type: {type(teacher_output)}")
    print(f"Teacher model output length: {len(teacher_output)}")
    for i, output in enumerate(teacher_output):
        if hasattr(output, 'shape'):
            print(f"Teacher output[{i}] shape: {output.shape}")
        else:
            print(f"Teacher output[{i}] type: {type(output)}")

# Forward pass for student model
with torch.no_grad():
    student_output = student_model.model(dummy_input)
    print(f"\nStudent model output type: {type(student_output)}")
    print(f"Student model output length: {len(student_output)}")
    for i, output in enumerate(student_output):
        if hasattr(output, 'shape'):
            print(f"Student output[{i}] shape: {output.shape}")
        else:
            print(f"Student output[{i}] type: {type(output)}")

# Forward pass for test model (yolov5n.yaml)
with torch.no_grad():
    test_output = test_model.model(dummy_input)
    print(f"\nTest model (yolov5n.yaml) output type: {type(test_output)}")
    print(f"Test model output length: {len(test_output)}")
    for i, output in enumerate(test_output):
        if hasattr(output, 'shape'):
            print(f"Test output[{i}] shape: {output.shape}")
        else:
            print(f"Test output[{i}] type: {type(output)}")

# Original training code (commented out)
# student_model.train(
#     data="visdrone.yaml",
#     teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
#     distillation_loss="cwd",
#     epochs=100,
#     batch=16,
#     workers=0,
#     imgsz=1024,
#     exist_ok=True,
#     name="distil_x_in_n"
# )