# 📰 Fake News Detection Model  
*A machine learning model to classify news articles as "fake" or "real".*

## 🚀 **Features**  
- **Text Preprocessing**: Cleans and tokenizes raw news text.  
- **TF-IDF Vectorization**: Converts text to numerical features.  
- **Logistic Regression Model**: Achieves **99% accuracy** (balanced dataset).  
- **Interactive Testing**: Predict labels for custom inputs.  

## ⚙️ Setup  
1. Install dependencies:  
   ```bash
   pip install pandas scikit-learn nltk joblib
   ```
2. Download the pre-trained model:  
   - `fake_news_detector_pipeline.pkl`  
   - `model_metadata.json` (optional)  

## 🧪Usage  
### Load and predict:  
```python
import joblib

# Load model
pipeline = joblib.load('fake_news_detector_pipeline.pkl')

# Predict
text = "Breaking: Moon landing was a hoax!"
prediction = pipeline.predict([text])[0]  # Output: 'fake' or 'real'
```

### Interactive testing (Jupyter/Colab):  
```python
from IPython.display import display, Markdown

text = input("Paste news text: ")
pred, confidence = predict_news(text)  # Uses saved model
print(f"Prediction: {pred} ({confidence*100:.0f}% confidence)")
```

## 📊 **Performance**  
| Metric       | Fake News | Real News |  
|--------------|-----------|-----------|  
| **Precision**| 99%       | 99%       |  
| **Recall**   | 99%       | 100%      |  

## 📂 **Files**  
- `train_model.ipynb`: Training notebook  
- `fake_news_detector_pipeline.pkl`: Saved model  
- `requirements.txt`: Python dependencies  

## ⚠️ **Limitations**  
- Works best on short-form news articles.  
- May need retraining for new topics/domains.  

---

### 🔗 **Quick Deploy**  
For API deployment (FastAPI example):  
```python
from fastapi import FastAPI  
app = FastAPI()  
app.include_router(prediction_router)  
```

---

Copy this into a `README.md` file in your project root. Customize links/commands as needed! 

