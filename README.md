

# Brain Tumor Detection using YOLOv10

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![YOLOv10](https://img.shields.io/badge/YOLOv10-%23FFD700.svg?style=for-the-badge&logo=yolo&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-%23FFB6C1.svg?style=for-the-badge&logo=gradio&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-%23FF69B4.svg?style=for-the-badge&logo=opencv&logoColor=black)
![Roboflow](https://img.shields.io/badge/Roboflow-%23007ACC.svg?style=for-the-badge&logo=roboflow&logoColor=white)

This project demonstrates how to use the YOLOv10 model for brain tumor detection using MRI images. The model has been trained and fine-tuned on a dataset to detect brain tumors efficiently.

---

## Table of Contents
- [Brain Tumor Detection using YOLOv10](#brain-tumor-detection-using-yolov10)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Tools and Libraries Used:](#tools-and-libraries-used)
  - [Installation and Setup](#installation-and-setup)
    - [YOLOv10 Installation](#yolov10-installation)
    - [Roboflow Integration](#roboflow-integration)
    - [Training the Model](#training-the-model)
  - [Model Inference](#model-inference)
  - [Visualization](#visualization)
  - [Web Application](#web-application)
  - [Results](#results)
  - [License](#license)
  - [References](#references)

---

## Overview

This project uses YOLOv10 for detecting brain tumors from MRI scans. We employ Roboflow for dataset management and Gradio for creating a web-based interface.

### Tools and Libraries Used:
- YOLOv10
- Roboflow
- Gradio
- Matplotlib
- OpenCV

---

## Installation and Setup

### YOLOv10 Installation

1. Install YOLOv10 from the official GitHub repository:

    ```bash
    !pip install -q git+https://github.com/THU-MIG/yolov10.git
    ```

   

2. Download the pre-trained YOLOv10n model weights:

    ```bash
    !wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt
    ```

    

3. Install Roboflow for dataset handling:

    ```bash
    !pip install -q roboflow
    ```

   

---

### Roboflow Integration

1. Install and set up Roboflow to manage and download the dataset:

    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="2A2n8FdjhCq768GBy1Yw")
    project = rf.workspace("brain-mri").project("mri-rskcu")
    version = project.version(3)
    dataset = version.download("yolov9")
    ```

    

---

### Training the Model

1. Start training the YOLOv10 model with the following command. You can modify parameters such as `epochs`, `batch`, and `model`:

    ```bash
    !yolo task=detect mode=train epochs=25 batch=32 plots=True \
    model=/content/yolov10n.pt \
    data=/content/MRI-3/data.yaml
    ```

    

---

## Model Inference

1. Load the trained model:

    ```python
    from ultralytics import YOLOv10

    model_path = "/content/runs/detect/train9/weights/best.pt"
    model = YOLOv10(model_path)
    ```

    
2. Run inference on the dataset with a confidence threshold:

    ```python
    result = model(source="/content/MRI-3/train/images", conf=0.25, save=True)
    ```

    

---

## Visualization

1. Visualize predictions from multiple images using Matplotlib:

    ```python
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    images = glob.glob("/content/runs/detect/predict/*.jpg")
    images_to_display = images[:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for i, ax in enumerate(axes.flat):
        if i < len(images_to_display):
            img = mpimg.imread(images_to_display[i])
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    ```

    

2. Visualize predictions on a single image:

    ```python
    result = model.predict(source="/content/MRI-3/valid/images/Tr-gl_0228_jpg.rf.b9ecef834d39f770e41b0585b63bdc1a.jpg", imgsz=640, conf=0.25)
    annotated_img = result[0].plot()
    annotated_img[:, :, ::-1]
    ```

    

---

## Web Application

1. Install Gradio:

    ```bash
    !pip install gradio
    ```

    

2. Define the `predict()` function and launch the app:

    ```python
    import gradio as gr
    import cv2
    import numpy as np

    def predict(image):
        result = model.predict(source=image, imgsz=640, conf=0.25)
        annotated_img = result[0].plot()
        annotated_img = annotated_img[:, :, ::-1]
        return annotated_img

    app = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label="Upload an image"),
        outputs=gr.Image(type="numpy", label="Detect Brain Tumor"),
        title="Brain Tumor Detection Using YOLOv10",
        description="Upload an image and the YOLOv10 model will detect and annotate the brain tumor."
    )

    app.launch()
    ```

    

---

## Results


---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

- [YOLOv10 Official GitHub Repository](https://github.com/THU-MIG/yolov10)
- [Roboflow](https://roboflow.com)
- [Gradio](https://gradio.app)

Feel free to contribute or suggest improvements!

---

**Note:** Replace the `images/` paths with the actual paths to the images used in your project. If you don't have these images, you might need to create or capture them based on your project's outputs.
