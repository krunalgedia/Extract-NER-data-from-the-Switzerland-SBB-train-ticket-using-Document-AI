import os
import re
import copy
import pandas as pd
#from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
try:
    from transformers import AutoTokenizer, LayoutLMForTokenClassification, AdamW
except ImportError:
    pass
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import torch
from tqdm import tqdm
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import pdf2image 
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

#C:\Program Files\Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(
    page_title = 'AI SBB ticket parser',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
    )
st.write("<style>div.Widget.stTitle {margin-top: -80px;}</style>", unsafe_allow_html=True)
st.title(':magenta[AI SBB train ticket parser]')
st.sidebar.header('User input')

image = st.sidebar.file_uploader("Upload SBB ticket", type=["pdf"])
button = st.sidebar.button("Parse ticket")

pages = []
if button and image is not None:
    if image.type == "application/pdf":
        images = pdf2image.convert_from_bytes(image.read())
        for page in images:
            pages.append(page)
            break

st.sidebar.write('Clear Database')
db_status = st.sidebar.button('Yes')
if db_status:
    os.remove('database.csv')


model = LayoutLMForTokenClassification.from_pretrained("KgModel/sbb_ticket_parser_LayoutLM")
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased",from_tf=False,from_pt=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def get_ocrdataset(image):
    width, height = image.size
    w_scale = 1000/width
    h_scale = 1000/height

    ocr_df = pytesseract.image_to_data(image, output_type='data.frame') \

    ocr_df = ocr_df.dropna() \
                   .assign(left_scaled = ocr_df.left*w_scale,
                           width_scaled = ocr_df.width*w_scale,
                           top_scaled = ocr_df.top*h_scale,
                           height_scaled = ocr_df.height*h_scale,
                           right_scaled = lambda x: x.left_scaled + x.width_scaled,
                           bottom_scaled = lambda x: x.top_scaled + x.height_scaled)

    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row) 
        actual_box = [x, y, x+w, y+h] 
        actual_boxes.append(actual_box)

    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))
    
    ocr_df['boxer'] = boxes
    ocr_df_ = pd.DataFrame({'box':[[i for i in ocr_df['boxer']]],'text':[[i for i in ocr_df['text']]] })
    dataset = Dataset(pa.Table.from_pandas(ocr_df_))

    return dataset

def process_ocrdataset(dataset):

    text = []

    #tokenizing, padding to expected seq length, and making final dataset with correct input col names
    def func(row, max_seq_length=512):
        token_boxes = []
        aligned_labels = []
        for word, box in zip(row['text'], row['box']):
            word_tokens = tokenizer.tokenize(word)
            text.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            #aligned_labels.append(label)
            #aligned_labels.extend([label for _ in range(len(word_tokens)-1)])

        special_tokens_count = 2
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        encoding = tokenizer(" ".join(row['text']), padding='max_length', truncation=True)

        pad_token_box = [0, 0, 0, 0]
        padding_length = max_seq_length - len(tokenizer(' '.join(row['text']), truncation=True)["input_ids"])
        token_boxes += [pad_token_box] * padding_length

        encoding['bbox'] = token_boxes

        return encoding

    dataset = dataset.map(func,remove_columns=["text", "box"])
    return dataset, text


def get_preddf(dataset, text, image):

    pred_labels = []
    for batch in dataset:
      outputs = model(
            input_ids=torch.Tensor(batch["input_ids"]).unsqueeze_(0).long().to(device),
            bbox=torch.Tensor(batch["bbox"]).unsqueeze_(0).long().to(device),
            attention_mask=torch.Tensor(batch["attention_mask"]).unsqueeze_(0).long().to(device),
            token_type_ids=torch.Tensor(batch["token_type_ids"]).unsqueeze_(0).long().to(device),
        )

      preds = torch.nn.functional.softmax(outputs.logits, dim=2).cpu().detach().numpy()
      preds_idx = preds.argmax(axis=2)[0]
      pred_labels.extend([model.config.id2label[idx] for idx in preds_idx])
      boxes = batch["bbox"]
      break

    pl = [pred_labels[i] for i,x in enumerate(dataset['bbox'][0]) if all([x != [0, 0, 0, 0], x != [1000, 1000, 1000, 1000]])]

    df = pd.DataFrame({'text':text,'label':pl})

    pred_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(pred_image)
    W, H = pred_image.size 
    bboxes = [unnormalize_box(box, W, H) for box in boxes]
    num_colors = len(model.config.id2label.keys())
    colors = plt.cm.get_cmap('tab10', num_colors)
    label_color_mapping = {label: to_hex(colors(i)) for i, label in enumerate(model.config.id2label.values())}

    for label, box in zip(pred_labels, bboxes):
        fill_color = label_color_mapping.get(label, 'black')
        draw.rectangle(box, outline='black')
        draw.text((box[0]+10, box[1]-10), text=label, fill=fill_color)

    return df, pred_image

def extract_text(input_text, part=None):
    text_parts      = re.findall(r'[a-zA-Z]+', input_text)
    number_parts    = re.findall(r'\d+', input_text)
    if part=='Date':
        number_parts = f'{number_parts[-2]}.{number_parts[-1]}.{number_parts[0]}' 
        return " ".join(text_parts),number_parts
    return " ".join(text_parts),".".join(number_parts)

def get_predrow(df):

    name     = " ".join(df.loc[df['label'].str.contains('TRAVELER'),'text'].values).replace(' ##','')
    cost     = " ".join(df.loc[df['label'].str.contains('COST'),'text'].values).replace(' ##','')
    ticketid = " ".join(df.loc[df['label'].str.contains('TICKET_ID'),'text'].values).replace(' ##','')
    date     = " ".join(df.loc[df['label'].str.contains('DATE'),'text'].values).replace(' ##','')
    ticket   = " ".join(df.loc[df['label'].str.endswith('TICKET'),'text'].values).replace(' ##','')

    #input_text = "gedia krunal bipin 24 . 01 . 1994"
    #input_text = 'chf 3 . 10'
    #input_text = 'ticket - id 001053058353'
    #input_text = '2023 ] 28 . 10 | 28 . 10 |'

    Name, DOB       = extract_text(name)
    _,Date          = extract_text(date,'Date')
    _,TicketID      = extract_text(ticketid)
    Ticketinfo,_    = extract_text(ticket)
    Currency,Amount = extract_text(cost) 

    new_row = {'Name': Name, 'DOB': DOB, 'Date of Travel': Date, 'Ticket ID': TicketID, 'Ticket Info': Ticketinfo, 'Amount': Amount, 'Currency': Currency}
    return new_row

if image and button:
    image = pages[0]
    dataset = get_ocrdataset(image)
    dataset, text = process_ocrdataset(dataset)
    pred_df, pred_image = get_preddf(dataset, text, image)
    new_row = get_predrow(pred_df)

    if not os.path.exists('database.csv'):
        df = pd.DataFrame(columns=['Name', 'DOB', 'Date of Travel', 'Ticket ID', 'Ticket Info', 'Amount', 'Currency'])     
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv('database.csv', index=False)
        st.header(':green[Updated Database]')
        st.table(df)
    else:
        df = pd.read_csv('database.csv')
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv('database.csv', index=False)
        st.header(':green[Updated Database]')
        st.table(df)
    

    col1, col2 = st.columns(2,gap='large')

    col1.header(':red[User SBB ticket]')
    col1.image(image, use_column_width=True) 

    col2.header(':violet[AI SBB ticket parsed]')
    col2.image(pred_image, use_column_width=True) 

else:
    st.header(':green[Sample Updated Database]')
    
    df = pd.read_csv('database_sample.csv')
    st.table(df)

    col1, col2 = st.columns(2,gap='large')
    
    col1.header(':red[Sample User SBB ticket]')
    col1.image('sample.jpg', use_column_width=True) 

    col2.header(':violet[AI SBB ticket parsed]')
    col2.image('sample_pred.jpg', use_column_width=True) 
