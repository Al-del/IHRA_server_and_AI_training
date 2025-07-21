# IHRA Server - Backend & AI Models

## 📝 Project Description
**IHRA (Instant Healthy Recipe App)** is a modern virtual assistant designed to simplify food decisions and promote a healthy lifestyle. The app allows users to take a photo of their fridge, analyzes the ingredients, and generates personalized recipes based on available items. It also provides health ratings for recipes, calorie information, and estimates the required exercise to burn the consumed calories.

This repository contains the Flask-based backend server and AI models that power the IHRA application.

## 🛠️ Technologies Used

### Backend
- **Flask** - Python web framework for handling API requests
- **PyTorch** - Deep learning framework for computer vision models
- **Hugging Face Transformers** - For NLP recipe generation
- **CatBoost** - For exercise duration regression
- **OpenCV** - Image processing

### AI Models
1. **Computer Vision (YOLOv8)**
   - Fine-tuned on 53 food ingredient classes
   - Optimized for GPU with batch size 4
   - Uses image augmentation (blur, median blur, CLAHE)
   - Trained with AdamW optimizer

2. **Natural Language Processing (Flan-T5-small)**
   - Fine-tuned with LoRA (only 0.45% parameters modified)
   - Trained on 100,000 recipes
   - Uses beam search with 5 beams
   - Limited to 200 output tokens

3. **Regression (CatBoostRegressor)**
   - Predicts exercise duration based on demographic/physiological features
   - Hyperparameter optimization with Optuna
   - Uses nested cross-validation (5 outer, 3 inner folds)
   - MAE of ~1.60 ± 0.003

## 📂 Repository Structure
```
IHRA_server_and_AI_training/
├── IHRA_serve/                      # Flask application
├── llm_fine_tunning/                 # LLM fine-tunning LoRA
├── object_seg/          # yolo training
├── reg_recepie/                # regression for workout
```

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Al-del/IHRA_server_and_AI_training.git
   cd IHRA_server_and_AI_training
   ```

2. **Download model weights**
   - Place YOLO weights in `app/models/computer_vision/weights/`
   - Place Flan-T5 weights in `app/models/nlp/weights/`
   - Place CatBoost model in `app/models/regression/weights/`

3. **Configure environment variables**
   Create a `.env` file based on `.env.example` and set your configuration.

4. **Run the server**
   ```bash
   flask run --host=0.0.0.0 --port=5000 (after run a tunneling service)
   ```

## 🌐 API Endpoints

### Computer Vision
- `POST /api/`
  - Input: Image file
  - Output: Detected ingredients with confidence scores

### Recipe Generation
- `POST /api/recepie`
  - Input: List of ingredients
  - Output: Generated recipe with steps

### Exercise Prediction
- `POST /api/workout`
  - Input: User attributes + calorie count
  - Output: Predicted exercise duration
