from ultralytics import YOLO


# 載入訓練好的模型
model = YOLO("best.pt")


# Train
results = model.train(data="config.yaml", epochs=50, batch=8)

# validation
val_metrics = model.val(data="config.yaml", split="val")

# test
test_metrics = model.val(data="config.yaml", split="test")

