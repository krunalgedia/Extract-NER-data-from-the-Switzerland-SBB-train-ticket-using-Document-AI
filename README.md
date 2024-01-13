# Extract NER data from the Switzerland's SBB train ticket using Document AI

Business Problem:

In the context of document processing, the manual extraction of relevant information from invoices poses a significant challenge, one that is poised to be automated in the near future. Presently, the verification of traveler and ticket details within invoices is conducted manually, particularly in larger firms dealing with a high volume of invoices on a daily basis. This manual parsing process demands considerable human resources, resulting in prolonged processing times and delays in reimbursement.

Project Objective:

The primary objective of this project is to address the inefficiencies associated with manual invoice parsing. Focusing specifically on SBB train tickets, we aim to implement an automated system that extracts pertinent information from the uploaded PDF documents. By leveraging automation, we intend to streamline the verification process and alleviate the burden on human resources.

Benefits

* Efficiency Gains: Automation reduces manual effort, leading to a significant reduction in processing time.
* Timely Reimbursement: Streamlining the verification process minimizes delays in reimbursement, enhancing overall financial efficiency.
* Scalability: As the system is designed to handle a large volume of invoices, it scales seamlessly with the growing needs of larger firms.

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

The goal of this project is to develop a real-time web application to
Key Features
* PDF Document Parsing: Automate the extraction of essential details from SBB train ticket PDFs.
* Relational Database Integration: Store the extracted information in a relational database for easy retrieval, verification and reimburesemt purpose.


## Installation

```bash
# Example installation command
pip install -r requirements.txt

# Run Web Application
streamlit run app.py
```

## Data

The data used consists of SBB train tickets for single and extension tickets. The training set consists of just 4 SBB train tickets. All tickets are in PDF form.

## Workflow
0. Prepare training data by annotation using the UBIAI tool [1]. This includes drawing bounding boxes and labeling in the BIOES tagging form [2].
1. Importing data
2. Observing data
3. Loading data in appropriate form.
4. Fine-tuning LayoutML model.
5. Preparing test set processing, including OCR of prediction documents using Pytesseract and getting the bounding box for all text in the test sample.
6. Running predictions on the bounding boxes of Pytesseract.
7. Update the database with relevant NER extracted from the model prediction on the annotated test sample.

* notebooks/SBB_TrainTicketParser.ipynb contains the end-to-end code for Document parsing with database integration.
* app.py contains the streamlit app code.

## Results

We fine-tuned using Facebook/Meta's LayoutLM (which utilizes BERT as the backbone and adds two new input embeddings: 2-D position embedding and image embedding) [3]. The model was imported from the Hugging Face library [4] with end-to-end code implemented in PyTorch. We leveraged the tokenizer provided by the library itself. For the test case, we perform the OCR using Pytesseract.

With just 4 SBB train tickets we can achieve an average F1 score of 0.81.   

| Epoch | Average Precision | Average Recall | Average F1 | Average Accuracy |
|--------:|------------:|---------:|-----:|-----------:|
|     145 |        0.89 |     0.77 | 0.82 |       0.9  |
|     146 |        0.9  |     0.79 | 0.84 |       0.9  |
|     147 |        0.86 |     0.77 | 0.81 |       0.89 |
|     148 |        0.87 |     0.78 | 0.82 |       0.9  |
|     149 |        0.86 |     0.77 | 0.81 |       0.89 |

The web application serves demo:
![Image 1](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/sample.gif) | ![Image 2](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/test1.gif)
--- | --- 
Opening page | Testing ... 

Once the user uploads the image, the document gets parsed and the information from the document gets updated in the relational database which can be used to verify the traveler's info and also to automate the travel cost-processing task.


## More ideas

Instead of using OCR from the UBIAI tool, it best is to use pyteserract or same OCR tool for train and test set. Further, with Document AI being developed at a rapid pace, it would be worthwhile to test newer multimodal models which hopefully either provide a new solution for not using OCR or inbuilt OCR since it is important to be consistent in preprocessing train and test set for best results.

Also, train on at least >50 tickets, since this was just a small test case to see how well the model can work.

## Dependencies

This project uses the following dependencies:

- **Python:** 3.10.12/3.9.18 
- **PyTorch:** 2.1.0+cu121/2.1.1+cpu
- **Streamlit:** 1.28.2 

- [SBB ticket parser model on Hugging Face](https://huggingface.co/KgModel/sbb_ticket_parser_LayoutLM)
  
## Contact

Feel free to reach out if you have any questions, suggestions, or feedback related to this project. I'd love to hear from you!

- **LinkedIn:** [Krunal Gedia](https://www.linkedin.com/in/krunal-gedia-00188899/)

## References
[1]: UBIAI annotation tool: [UBIAI](https://app.ubiai.tools/Projects).

[2]: Named Entity Recognition, Vijay Krishnan and Vignesh Ganapathy: [NER](http://cs229.stanford.edu/proj2005/KrishnanGanapathy-NamedEntityRecognition.pdf) 

[3]: LayoutLM: Pre-training of Text and Layout for Document Image Understanding: [LayoutLM] (https://arxiv.org/abs/1912.13318)

[4]: [Huggingface LayoutLM] (https://huggingface.co/docs/transformers/model_doc/layoutlm)


