from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from scipy.special import softmax
import easyocr


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

        if scores[0] > 0.5:
            return Response({"status": "ok", "comment": req.POST.get("comment")}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "not ok", "comment": req.POST.get("comment")}, status=status.HTTP_200_OK)


@api_view(['POST'])
def evaluate_image(req):
    if req.method == 'POST':
        evaluator = evaluator_api.utils.load_eval_model.get_img_eval_model()

        tokenizer = evaluator["tokenizer"]
        model = evaluator["model"]

        img = req.FILES("image")
        reader = easyocr.Reader(['en'])
        results = reader.readtext(img)
        text = ''
        for result in results:
            text += result[1] + ' '

        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        if scores[0] > 0.5:
            return Response({"status": "ok", "comment": req.POST.get("comment")}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "not ok", "comment": req.POST.get("comment")}, status=status.HTTP_200_OK)
