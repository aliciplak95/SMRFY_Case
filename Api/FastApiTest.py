from typing import Optional
import json
from fastapi import FastAPI, Form
from pydantic import BaseModel
from simpletransformers.classification import MultiLabelClassificationModel
import torch
from fastapi.middleware.cors import CORSMiddleware


class Item(BaseModel):
    message: str


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=[""],
)


@app.post("/items/")
async def create_item(item: Item):
    print(item.message)

    mod = MultiLabelClassificationModel(
        'bert', 'outputs/', args={}, use_cuda=False)

    predictions, raw_outputs = mod.predict([item.message])

    response = "null"
    if predictions[0][0] == 1:
        response = "hesap işlemi"
    elif predictions[0][1] == 1:
        response = "iade"
    elif predictions[0][2] == 1:
        response = "iptal"
    elif predictions[0][3] == 1:
        response = "kredi"
    elif predictions[0][1] == 1:
        response = "kredi kartı"
    elif predictions[0][5] == 1:
        response = "musteri hizmetleri"
    else:
        response = "tanımlanamayan istek"

    print(response)

    return response
