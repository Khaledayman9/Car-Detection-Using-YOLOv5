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
  - It was trained for 50 epoch.
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
  - Box Loss: 0.03753
  - Objectness Loss: 0.01585
  - Classification Loss: 0.00000
  - Class: all
  - Images: 71
  - Instances: 101
  - Precision (P): 0.967
  - Recall (R): 0.950
  - Mean Average Precision at IoU=0.5 (mAP50): 0.977
  - Mean Average Precision (mAP): 0.620
- The GPU Memory Usage was 2.45G.


# Acknowledgments
We would like to acknowledge the YOLOv5[^2] repository for its invaluable contribution to the object detection training process. The YOLOv5 architecture and its associated tools provided the foundation and functionality necessary to develop and train our car detection model efficiently. We appreciate the efforts and open-source contributions of the YOLOv5 team, which have significantly enhanced the performance and capabilities of our project.

[^2]: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

