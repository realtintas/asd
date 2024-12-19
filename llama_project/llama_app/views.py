from django.shortcuts import render
from django.http import JsonResponse
from .utils.model_loader import LlamaModelHandler

# Model yolunu ayarlayın
MODEL_PATH = "Llamaresul/original/consolidated.00.pth"
PARAM_PATH = "Llamaresul/original/params.json"
TOKENIZER_PATH = "Llamaresul/original/tokenizer.model"

# Model Handler sınıfını oluştur
model_handler = LlamaModelHandler(MODEL_PATH, PARAM_PATH, TOKENIZER_PATH)

def predict_text(request):
    if request.method == "POST":
        user_input = request.POST.get("input_text", "")
        if user_input:
            result = model_handler.predict(user_input)
            return JsonResponse({"predicted_text": result})
        return JsonResponse({"error": "Input not provided"})

    return JsonResponse({"message": "Send a POST request with input_text"})
