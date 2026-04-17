from django.http import JsonResponse
from django.shortcuts import render

from .ml import get_model_metrics, predict_language, translate_text


def home(request):
    return render(request, "index.html")


def predict(request):
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        predicted_lang = predict_language(text) if text else None
        return render(
            request,
            "predict.html",
            {"text": text, "predicted_lang": predicted_lang},
        )
    return render(request, "predict.html")


def translate(request):
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        target_lang = request.POST.get("target_lang", "en")
        predicted_lang = predict_language(text) if text else None
        translated_text = translate_text(text, target_lang) if text else ""
        return render(
            request,
            "translate.html",
            {
                "text": text,
                "target_lang": target_lang,
                "predicted_lang": predicted_lang,
                "translated_text": translated_text,
            },
        )
    return render(request, "translate.html")


def model_accuracy(request):
    return JsonResponse(get_model_metrics())
