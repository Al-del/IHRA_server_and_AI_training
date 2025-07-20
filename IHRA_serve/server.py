import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import predict_bbox, predict_recipe
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import traceback
import joblib
import pandas as pd
workout_pipeline = joblib.load("./workout_pred_pipe.pkl")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "DavidGI23200/recepie_llm_fine_tuned_with_lora"
tokenizer_recepie = AutoTokenizer.from_pretrained(model_name)
model_recepie = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

model = YOLO("./best.pt")

app = Flask(__name__)
CORS(app)

SPOON_KEY = os.getenv("SPOONACULAR_KEY", "3ba7c23eb4a14ea0a5ed0682cb0ab123")
INGREDIENTS_URL = "https://api.spoonacular.com/recipes/findByIngredients"
RECIPE_INFO_URL = "https://api.spoonacular.com/recipes/{id}/information"

@app.route('/', methods=['POST'])
def receive_data():
    data = request.get_json()
    base64_img = data.get('base64_img', '')

    if not base64_img:
        return jsonify({'status': 'error', 'message': 'No image data received.'}), 400

    try:
        _, encoded = base64_img.split(',', 1) if ',' in base64_img else ('', base64_img)
        img_data = base64.b64decode(encoded)

        ingred_list = predict_bbox(model, img_data)

        if not ingred_list:
            print("No ingredients detected.")
            return jsonify({'status': 'success', 'ingredients': [], 'recipes': []}), 200

        print("Detected ingredients:", ingred_list)

        params = {
            'apiKey': SPOON_KEY,
            'ingredients': ','.join(ingred_list),
            'number': 10,
            'ranking': 1,
            'ignorePantry': True
        }

        resp = requests.get(INGREDIENTS_URL, params=params)
        resp.raise_for_status()
        recipes = resp.json()

        results = []
        for recipe in recipes:
            recipe_id = recipe.get('id')
            title = recipe.get('title')
            image_data = recipe.get('image')

            img_url = None
            if image_data:
                if image_data.startswith('http'):
                    img_url = image_data
                elif '/' in image_data:
                    img_url = f"https://spoonacular.com/recipeImages/{image_data}"
                else:
                    img_url = f"https://spoonacular.com/recipeImages/{recipe_id}-{image_data}"

            # Fetch detailed recipe info with nutrition data
            try:
                info_resp = requests.get(
                    RECIPE_INFO_URL.format(id=recipe_id),
                    params={
                        'apiKey': SPOON_KEY,
                        'includeNutrition': 'true'  # Ensure nutrition data is included
                    }
                )
                info_resp.raise_for_status()
                info_data = info_resp.json()

                # Extract ingredients
                full_ingredients = [
                    ing.get('original') or ing.get('originalString') or ing.get('name') 
                    for ing in info_data.get('extendedIngredients', [])
                ]

                # Extract calories
                calories = None
                nutrition = info_data.get('nutrition')
                if nutrition:
                    for nutrient in nutrition.get('nutrients', []):
                        if nutrient.get('name', '').lower() == 'calories':
                            amount = nutrient.get('amount', 0)
                            unit = nutrient.get('unit', 'kcal')
                            calories = f"{round(amount)} {unit}" if amount else None
                            break

                results.append({
                    'id': recipe_id,
                    'title': title,
                    'image': img_url,
                    'ingredients': full_ingredients,
                    'calories': calories,
                    'readyInMinutes': info_data.get('readyInMinutes'),
                    'servings': info_data.get('servings')
                })

            except requests.exceptions.RequestException as e:
                print(f"Failed to get details for recipe {recipe_id}: {str(e)}")
                continue  # Skip this recipe but continue with others

        return jsonify({
            'status': 'success',
            'ingredients': ingred_list,
            'recipes': results
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({'status': 'error', 'message': f"API request failed: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/recipe', methods=['POST'])
def generate_recipe():
    data = request.get_json()
    title = data.get('title', '')
    ingredients = data.get('ingredients', [])

    if not title or not ingredients:
        return jsonify({'status': 'error', 'message': 'Missing title or ingredients'}), 400

    try:
        title_ingredients = {
            'title': title,
            'ingredients': ', '.join(ingredients)
        }
        print("[INFO] Input for recipe generation:", title_ingredients)

        recipe_text = predict_recipe(title_ingredients, tokenizer_recepie, device, model_recepie)
        print("[INFO] Generated recipe:", recipe_text)

        return jsonify({
            'status': 'success',
            'generated_recipe': recipe_text
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
@app.route('/workout', methods=['POST'])
def handle_workout():
    try:
        data = request.get_json()
        print("[/workout] Received JSON data:", data)
        pred = {}
        pred["Sex"] = data["gender"]
        pred["Age"] = data["age"]
        pred["Height"] = data["height"]
        pred["Weight"] = data["weight"]
        pred["Heart_Rate"] = data["heartRate"]
        pred["Body_Temp"] = data["bodyTemp"]
        pred["Calories"] = data["calories"]
        pred = pd.DataFrame([pred])
        time = workout_pipeline.predict(pred)
        print(time[0])
        return jsonify({
            'status': 'success',
            'message': f"Estimated workout time is {float(time[0]):.1f} minutes"
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)