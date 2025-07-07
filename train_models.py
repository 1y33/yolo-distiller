from ultralytics import YOLO
import torch
import ultralytics
import ultralytics.utils
import ultralytics.utils.torch_utils


MODEL_CONFIG = "yolov5x.yaml"
MODEL_WEIGHTS = "yolov5x.pt"

TRAINING_CONFIG = {
    "data": "visdrone.yaml",
    "resume": False,
    "batch": 0.60,
    "epochs": 100,
    "dropout": 0.0,
    "lr0": 1e-3,
    "lrf": 1e-5,
    "name": "Test",
    "imgsz": 1024,
    "cos_lr": True,
    "seed": 0,
    "optimizer": "Adam",
    "save_period": 10,
}

EXPORT_CONFIG = {
    "format": "edgetpu",
    "imgsz": 1024
}

INFERENCE_CONFIG = {
    "test_image": "images/test_image.jpeg",
    "input_shape": [1, 3, 1024, 1024]
}


def train(use_pretrained=False, config_override=None):
    print("Starting training...")
    
    if use_pretrained:
        model = YOLO(MODEL_CONFIG).load(MODEL_WEIGHTS)
        print(f"Loaded model from {MODEL_CONFIG} with weights {MODEL_WEIGHTS}")
    else:
        model = YOLO(MODEL_CONFIG)
        print(f"Loaded model from {MODEL_CONFIG}")
    
    train_config = TRAINING_CONFIG.copy()
    if config_override:
        train_config.update(config_override)
    
    print(f"Training config: {train_config}")
    model.train(**train_config)
    print("Training completed!")
    
    return model


def main():
    train(use_pretrained=True)  # Train with pre-trained weights
    # train(use_pretrained=False)  # Train from scratch
    
    
    # inference(image_path=INFERENCE_CONFIG["test_image"])  # Image inference
    # inference(model_path=MODEL_CONFIG,tensor_inference=True)  # Tensor inference
    
    print("Workflow completed!")

if __name__ == "__main__":
    main()