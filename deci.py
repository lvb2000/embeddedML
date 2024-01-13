import super_gradients

yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco")
model_predictions  = yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg")

prediction = model_predictions[0].prediction
bboxes = prediction.bboxes_xyxy
print(bboxes)
class_labels = prediction.labels
print(class_labels)
confidences =  prediction.confidence.astype(float)
print(confidences)

for bbox, class_label, confidence in zip(bboxes, class_labels, confidences):
    # class name '0' equals persons
    if class_label == 0:
        if confidence > 0.5:
            print(f"Found Person with confidence {confidence} at {bbox}")