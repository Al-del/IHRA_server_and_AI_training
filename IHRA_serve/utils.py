import cv2
import numpy as np

def predict_bbox(model, img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    ingredients = []
    for result in results:
        names = result.names
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # numpy array of class indices
        detected_names = [names[cid] for cid in class_ids]
        ingredients.extend(detected_names)

    ingredients = list(set(ingredients))

    return ingredients
def predict_recipe(title_ingredients, tokenizer, device, model):
    """
    Generate recipe directions from title and ingredients.
    Format expected by the model: 'Title: <title> Ingredients: <comma-separated ingredients>'
    """
    # Correct format
    text = f"Title: {title_ingredients['title']} Ingredients: {title_ingredients['ingredients']}"
    
    # Tokenize and generate
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1000,
        num_beams=5,
        early_stopping=True
    )
    
    # Decode and return
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    start = decoded.find("directions")
    return decoded[start:] if start != -1 else decoded