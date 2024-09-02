# Car-Detection-Using-YOLOv5
Detecting cars within images. The primary goal is to develop and implement a car detection system capable of identifying and drawing bounding boxes around cars in various traffic scenes. The system is designed to work with images containing multiple vehicles and to accurately localize each car with bounding boxes.

# Dataset
The dataset[^1] consists of images of traffic scenes where each image may contain multiple cars. These images are provided along with corresponding bounding box annotations for training and evaluation. The dataset is organized into training, validation, and test directories, with labels formatted for YOLO-based object detection models. The dataset contains media of cars in all views.

[^1]: [Car Object Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)

# Objectives
- **Car Detection:** Build and train a model to detect cars in images. The model should be able to handle various traffic scenarios and accurately identify cars with bounding boxes.
- **Visualization:** Implement functionality to visualize the detection results by drawing bounding boxes around detected cars in the images.
- **Handling Large Outputs:** Ensure that the visualization function efficiently handles large images and displays the results in a clear and interpretable manner.

# Approach
## Data Preprocessing
- Data Cleaning: Ensure the dataset is free from corrupted or unreadable images. Handle any issues by verifying the integrity of images before proceeding with training.
- Directory Structure: Organize the dataset into appropriate directories for training, validation, and testing. The typical structure includes:
  - train/images/: Contains training images.
  - train/labels/: Contains YOLO format label files for training images.
  - val/images/: Contains validation images.
  - val/labels/: Contains YOLO format label files for validation images.
  - test/images/: Contains test images.
- Label Formatting: Convert annotations into YOLO format, which includes converting bounding box coordinates into normalized values relative to image dimensions. Each label file contains one line per object with the format: class_id x_center y_center width height. 

## Model Training
- YOLO-Based Model: Utilize the YOLO (You Only Look Once) architecture for object detection. The model is trained to specifically identify cars based on the provided dataset. YOLO’s capability to detect multiple objects in a single image makes it suitable for this task.
- The hyperparameter were:
  - It was trained for 100 epoch.
  - Images size was 640x640.
  - Batch size was 16.

## YAML File Formation
- Create a YAML Configuration File: Define paths for training, validation, and testing datasets in a YAML file used by YOLO for training. The YAML file should include:
  - train: Path to the training images directory.
  - val: Path to the validation images directory.
  - test: Path to the testing images directory.
  - nc: Number of classes (e.g., 1 for cars).
  - names: List of class names (e.g., ['car']).
- Example YAML configuration:
```yaml
train: data/train/images
val: data/val/images
test: data/test/images
nc: 1
names: ['car']
```

## Prediction
Run Predictions: Develop a function to run predictions on test images. This function processes each image using the trained model and returns predictions, including bounding boxes around detected cars.

## Visualization
- Draw Bounding Boxes: Implement a function to draw bounding boxes around detected cars. The function should resize images to fit within a displayable size and ensure that the detected cars are clearly highlighted.
- Handle Large Images: Ensure the visualization function can manage large images effectively by resizing them before plotting to avoid excessive memory usage and display issues.

## Testing on External Samples
- Test on Samples from the Internet: Extend testing to include images from the internet or other sources outside the test set. This helps evaluate the model's performance on diverse, unseen data and ensures its generalizability.
- Compare Predictions: Analyze the predictions on external samples to understand how well the model adapts to different contexts and scenarios. Adjustments to the model or additional training data may be necessary based on these results.

## Error Handling
Manage Image Errors: Include error handling to manage issues such as corrupted or unreadable image files. Ensure the system gracefully skips or handles these problematic images without interrupting the processing of other images.


# Results
- After the training process, the model achieved the following results:
  - Box Loss: 0.02876
  - Objectness Loss: 0.01235
  - Classification Loss: 0 (indicating no classification error, because only one class is detected)
  - Class: all
  - Images: 71
  - Instances: 108
  - Precision (P): 0.989 (98.9%)
  - Recall (R): 0.963 (96.3%)
  - Mean Average Precision at IoU 0.5 (mAP50): 0.989 (98.9%)
  - Mean Average Precision at IoU 0.5:0.95 (mAP50-95): 0.658 (65.8%)
- These statistics suggest that the model is performing very well in terms of precision and recall, with high mAP50, indicating good detection performance for the object class in your test dataset. The mAP50-95 score, which is lower, suggests that there may still be room for improvement in detecting objects across varying IoU thresholds.
- The GPU Memory Usage was 2.45G.
- Confidence Threshold: This value sets the minimum confidence level that a detection must have to be considered valid. In this case, only detections with a confidence score of 0.4 or higher will be considered as valid detections. Lower confidence detections will be ignored.
## Evaluating on Test Set:
- The model performed good on the test dataset:
  
![Predictions](https://github.com/user-attachments/assets/0247c3de-20ae-48bf-9f7f-ad4ba67fa366)

- To showcase this, 15 random samples from the test dataset are plotted, Here are 5 of them:
  
![S1](https://github.com/user-attachments/assets/cbf6b350-d15a-44d7-8128-31474e025c47)

![S2](https://github.com/user-attachments/assets/34a6d9ad-a6aa-4699-a7b7-f13f9d75f921)

![S3](https://github.com/user-attachments/assets/b8799ea0-7ab6-4e9d-a178-f81ce91f314f)

![S4](https://github.com/user-attachments/assets/85e8b101-4909-4fe8-a8f7-fea3277bc7fc)

![S5](https://github.com/user-attachments/assets/5becc06c-c6c2-4a64-a43d-aff258af6cd4)

- And finally, 6 images outside the dataset are tested by the model to examine the model's performance:

![SS1](https://github.com/user-attachments/assets/4fae7c04-8d0b-4f52-a9a8-2dc24952e9d6)

![SS2](https://github.com/user-attachments/assets/6b5fcb21-e314-4711-94a8-466f2323dd03)

![SS3](https://github.com/user-attachments/assets/791b33f1-5a58-4fd8-b4ba-118845235ece)

![SS4](https://github.com/user-attachments/assets/5301fc5c-ae09-4888-962a-d6582fcea38c)

![SS5](https://github.com/user-attachments/assets/15604ba2-9d05-4fb5-b108-c45d1dd1439d)

![SS6](https://github.com/user-attachments/assets/bec9a468-ebac-40f1-9e3b-c9d034c019e3)

- You can notice in the image "4.jpg" that an incorrect bounding box was predicted. This indicates that the model could benefit from further improvement.

# Conclusion
The current YOLO-based car detection model shows promising results with high precision and recall rates. However, as observed in some test images, there are instances of incorrect bounding box predictions, indicating that the model still has room for improvement. Potential enhancements could involve further tuning of hyperparameters, augmenting the dataset with more diverse examples, or exploring more advanced model architectures. Continued development and testing are essential to achieve more accurate and reliable detections, especially in challenging scenarios. The contributions from the YOLOv5 repository have been instrumental in advancing this project, and ongoing refinement will help in creating a more robust solution for car detection tasks.

# Acknowledgments
We would like to acknowledge the YOLOv5[^2] repository for its invaluable contribution to the object detection training process. The YOLOv5 architecture and its associated tools provided the foundation and functionality necessary to develop and train our car detection model efficiently. We appreciate the efforts and open-source contributions of the YOLOv5 team, which have significantly enhanced the performance and capabilities of our project.

[^2]: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

# Technologies
- Python
- Kaggle Notebook

