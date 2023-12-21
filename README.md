# SBB_TrainTicketParser

Business Problem
In the context of document processing, the manual extraction of relevant information from invoices poses a significant challenge, one that is poised to be automated in the near future. Presently, the verification of traveler and ticket details within invoices is conducted manually, particularly in larger firms dealing with a high volume of invoices on a daily basis. This manual parsing process demands considerable human resources, resulting in prolonged processing times and delays in reimbursement.

Project Objective
The primary objective of this project is to address the inefficiencies associated with manual invoice parsing. Focusing specifically on SBB train tickets, we aim to implement an automated system that extracts pertinent information from the uploaded PDF documents. By leveraging automation, we intend to streamline the verification process and alleviate the burden on human resources.

Benefits
- Efficiency Gains: Automation reduces manual effort, leading to a significant reduction in processing time.
- Timely Reimbursement: Streamlining the verification process minimizes delays in reimbursement, enhancing overall financial efficiency.
- Scalability: As the system is designed to handle a large volume of invoices, it scales seamlessly with the growing needs of larger firms.

Through the successful implementation of this project, we aim to contribute to the optimization of invoice processing workflows, ultimately fostering efficiency, accuracy, and timely reimbursement within larger organizations.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Workflow](#workflow)
- [Results](#results)
- [More ideas](#More-ideas)
- [Dependencies](#dependencies)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Project Overview

[Provide a brief overview of your project, including its purpose, goals, and any relevant context.]
The goal of this project is to develop a real-time web application to
Key Features
* PDF Document Parsing: Automate the extraction of essential details from SBB train ticket PDFs.
* Relational Database Integration: Store the extracted information in a relational database for easy retrieval, verification and reimburesemt purpose.


## Installation

[Include instructions for installing any necessary dependencies or setting up the environment. You may want to provide code snippets or refer to a separate document for detailed installation steps.]

```bash
# Example installation command
pip install -r requirements.txt

# Run Web Application
streamlit run app.py
```

## Data

The data used consists of SBB train tickets for single and extension tickets. Training set consists of just 4 SBB train tickets. They are all in PDF form,

## Workflow
1. Importing data
2. Observing data
3. Loading data in appropriate form.
4. Training different models and comparing metrics.
5. Running predictions on the model
6. Running Grad CAM insights on the final attention layer

* notebooks/SBB_TrainTicketParser.ipynb contains the end-to-end code for Document parsing with database integration.
* app.py contains the streamlit app code.

## Results

For the first task of image classification, we tried Transfer learning using popular CNN models like VGG19, ResNet, and EfficientNet as well as Vision models as Vision Transformer. Using average recall of Benign and Malignant tumors, we found EfficientNet to be the best-performing one on the test set. The models were implemented in tf-keras with callbacks implemented for overtraining.
The following table gives the gist of the performance.

| model           |   Precision_Normal |   Precision_Benign |   Precision_Malignant |   Recall_Normal |   Recall_Benign |   Recall_Malignant |   Recall_BM |
|:----------------|-------------------:|-------------------:|----------------------:|----------------:|----------------:|-------------------:|------------:|
| EfficientNetB7  |               0.72 |               0.86 |                  0.82 |            0.85 |            0.83 |               0.78 |        0.81 |
| ResNet152V2     |               0.71 |               0.89 |                  0.86 |            0.89 |            0.88 |               0.73 |        0.8  |
| VGG19           |               0.74 |               0.86 |                  0.62 |            0.93 |            0.73 |               0.73 |        0.73 |
| EfficientNetV2S |               0.53 |               0.85 |                  0.69 |            0.89 |            0.69 |               0.66 |        0.68 |
| ConvNeXtBase    |               0.7  |               0.84 |                  0.7  |            0.78 |            0.8  |               0.73 |        0.76 |
| vit_b16         |               0.27 |               0.75 |                  0.45 |            0.81 |            0.07 |               0.73 |        0.4  |

The web app gives the prediction using the EfficientNet model. Following is the demo for each of the three classes:
![Image 1](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/sample.gif) | ![Image 2](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/test.gif)
--- | --- 
Opening page | Testing ... 


For the second task of image semantic segmentation, we fine-tuned the pre-trained models used in the medical literature like UNet, ResUNet, ResUNet with Attention. Using dice score and Intersection over Union (IoU) as a metric, all the models were found to be performing similarly. We finally decided to go with ResUNet with Attention since it provided negligible gains in metrics over the other two models. The models were implemented in tf-keras.

The web app gives the segmentation prediction using the EfficientNet model. Following is the demo for each of the three classes:
![Image 1](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/benign.gif) | ![Image 2](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/malignant.gif) | ![Image 3](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/normal.gif)
--- | --- | ---
Benign Scan | Malignant Scan | Normal Scan

For the third task, we used Grad-CAM which uses Class Activation Map along with gradient information to give insights into the model. The final heatmap is overlaid on the input image as shown in the web app for both classification and segmentation predictions.

<img src="https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/difficult.gif" alt="Image" width="400"/> Such Grad-CAM heatmaps can help greatly in giving insights to the user about the model's focus in decision-making. This can help in cases like the one shown on the left side where the segmentation prediction is not the best. However, looking at the GradCAM heatmap shows the areas the model focussed on and we see the focus included the region of the tumor but the model. Thus, even though the model fails here, such insights can help the doctor to detect the probable areas of the tumor.

## More ideas

For the classification model, maybe cross combination or concatenating transformer vision model with CNN models could work even better since transformer models are known to capture the global picture while CNN models look more at the local picture. 

For the segmentation model, maybe stacking of models could be tried.

## Dependencies

This project uses the following dependencies:

- **Python:** 3.10.12
- **Tensorflow:** 2.15.0
- **Streamlit:** 1.28.2 

- [Trained Classification model](https://www.dropbox.com/scl/fi/lnp23cdyo4eq0nckn2vtq/Detection_model?rlkey=ll5wy8fhw4mb9fopk83xdkbn0&dl=0)
- [Trained Segmentation model](https://www.dropbox.com/scl/fi/cq1tescfxr1uvp8z8fb2g/Segmentation_model?rlkey=s81dvvzzdlj0q92kjxvno78vr&dl=0)
  
## Contact

Feel free to reach out if you have any questions, suggestions, or feedback related to this project. I'd love to hear from you!

- **LinkedIn:** [Krunal Gedia](https://www.linkedin.com/in/krunal-gedia-00188899/)

## References
[1]: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863


