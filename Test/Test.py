from simpletransformers.classification import MultiLabelClassificationModel
import torch


def main():

    mod = MultiLabelClassificationModel(
        'bert', 'outputs/', args={}, use_cuda=False)

    text = input("Sınıf aranacak metni giriniz : ")
    predictions, raw_outputs = mod.predict([text])

    print(predictions)

    return predictions


if __name__ == '__main__':
    main()


predictions = main()
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
