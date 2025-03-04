# 🍔 Food Image Classification Web App (Deep Learning)

This project is a **Food Image Classification** web application built using **Streamlit** and **TensorFlow**. The app can classify images of 101 different food categories using the **EfficientNetB0** deep learning model. Users can upload images via three methods:

1. **File Upload**
2. **Camera Input**
3. **Image URL**
4. This Deep Learning model is trained on the [FOOD101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101)
5. Check the Code of the model on my [GitHub](https://github.com/abhi24112/Streamlit_app_food101)
6. My Portfolio : [Abhishek Portfolio](https://abhishek-portfolio-tau.vercel.app/)

---

## 🚀 Features

✅ **Multi-Input Support:**
   - Upload food images via file, camera, or URL.

✅ **Fast and Accurate Predictions:**
   - Uses a pre-trained **EfficientNetB0** model with 81% accuracy.

✅ **Interactive UI:**
   - Built with **Streamlit** for a smooth and responsive experience.

✅ **Real-time Image Processing:**
   - Seamless image handling and live predictions.

✅ **Automatic Model Caching:**
   - Efficient model loading using `st.cache_resource()` for faster execution.

---

## 📊 Tech Stack

- **TensorFlow**: For building and loading the EfficientNetB0 model
- **Streamlit**: For creating the interactive web application
- **Pandas/Numpy**: For data manipulation
- **Pillow**: For image processing
- **Requests**: For handling image input from URLs
- **Seaborn & Matplotlib**: For future visualization support

---

## 📂 Project Structure

```
Food101-WebApp/
├── app.py              # Main application script
├── efficientnetb0_fine_tuned_101_classes_mixed_precision.keras  # Pre-trained model
├── requirements.txt    # Dependencies
└── README.md           # Project Documentation
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
https://github.com/abhi24112/Streamlit_app_food101.git
cd Streamlit_app_food101
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web App

```bash
streamlit run app.py
```

---

## 📦 Deployment

This app can be easily deployed on **Streamlit Community Cloud** or **Heroku**.

1. Ensure `app.py` and `requirements.txt` are in the root directory.
2. Push your code to a GitHub repository.
3. Connect your repository to Streamlit Cloud.

---

## 📝 Usage Guide

1. **Choose Input Method:**
   - Upload an image via file uploader.
   - Capture a live image using your device camera.
   - Provide a direct image URL.

2. **View Prediction:**
   - The app will display the predicted food category and the prediction confidence.

---

## 📜 requirements.txt

```
numpy
pandas
matplotlib
seaborn
Pillow
tensorflow
streamlit
requests
```

---

## 🤝 Contributing

Feel free to open issues and submit pull requests! All contributions are welcome.

---

## 📧 Contact

For queries or suggestions, reach out at: **abhishek9910k@gmail.com**

---

⭐ **If you found this project helpful, consider giving it a star on GitHub!** ⭐

