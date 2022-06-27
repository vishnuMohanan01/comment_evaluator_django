import os
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from scipy.special import softmax
import pytesseract
from PIL import Image


# Create your views here.
import evaluator_api.utils.load_eval_model


@api_view(['POST'])
def evaluate_text(req):
    if req.method == 'POST':
        evaluator = evaluator_api.utils.load_eval_model.get_text_eval_model()

        tokenizer = evaluator["tokenizer"]
        model = evaluator["model"]

        text = req.POST.get("comment")

        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        if 0.45 <= scores[0] <= 0.55:
            return Response({"status": "ok", "comment": req.POST.get("comment"), "rating": 3}, status=status.HTTP_200_OK)

        elif scores[0] > 0.55:
            """Not Offensive
            get confidence value in scores[0] 
            """

            if scores[0] > 0.75:
                return Response({"status": "ok", "comment": req.POST.get("comment"), "rating": 5}, status=status.HTTP_200_OK)
            elif scores[0] < 0.75:
                return Response({"status": "ok", "comment": req.POST.get("comment"), "rating": 4}, status=status.HTTP_200_OK)

        else:
            """Offensive
            get confidence value in scores[1] 
            """

            if scores[1] > 0.75:
                return Response({"status": "not ok", "comment": req.POST.get("comment"), "rating": 1}, status=status.HTTP_200_OK)
            elif scores[1] < 0.75:
                return Response({"status": "not ok", "comment": req.POST.get("comment"), "rating": 2}, status=status.HTTP_200_OK)


@api_view(['POST'])
def evaluate_image(req):
    if req.method == 'POST':
        evaluator = evaluator_api.utils.load_eval_model.get_img_eval_model()

        tokenizer = evaluator["tokenizer"]
        model = evaluator["model"]

        img = req.FILES["image"]
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        uploaded_file_url = fs.url(filename)
        print("uploaded URL: ", uploaded_file_url)

        file_path = os.getcwd() + uploaded_file_url
        print("File Path: " + file_path)

        text = pytesseract.image_to_string(Image.open(file_path))
        text = text.replace('\n', ' ').replace('\r', ' ')

        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        if 0.45 <= scores[0] <= 0.55:
            return Response({"status": "ok", "comment": text, "rating": 3}, status=status.HTTP_200_OK)

        elif scores[0] > 0.55:
            """Not Offensive
            get confidence value in scores[0] 
            """

            if scores[0] > 0.75:
                return Response({"status": "ok", "comment": text, "rating": 5}, status=status.HTTP_200_OK)
            elif scores[0] < 0.75:
                return Response({"status": "ok", "comment": text, "rating": 4}, status=status.HTTP_200_OK)

        else:
            """Offensive
            get confidence value in scores[1] 
            """

            if scores[1] > 0.75:
                return Response({"status": "not ok", "comment": text, "rating": 1}, status=status.HTTP_200_OK)
            elif scores[1] < 0.75:
                return Response({"status": "not ok", "comment": text, "rating": 2}, status=status.HTTP_200_OK)
