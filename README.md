# 🎬 IMDB Movie Review Sentiment Analysis using Simple RNN

This project is an end-to-end **Sentiment Analysis Web App** built using **TensorFlow/Keras** and **Streamlit**. It uses a pre-trained **Simple RNN** model to classify movie reviews from the IMDB dataset as either **Positive** or **Negative**.

---

## 📌 Features

- ✅ Trained on IMDB reviews using TensorFlow's built-in dataset  
- ✅ Uses \`Embedding\` + \`SimpleRNN\` layers for sequence modeling  
- ✅ Streamlit web interface for real-time review classification  
- ✅ Shows confidence score with sentiment output  
- ✅ Preprocessing pipeline matches training time setup  

---

## 🧠 Model Architecture

- **Input Layer:** Padded sequence of word indices (maxlen=500)  
- **Embedding Layer:** Converts words to 64-dimensional dense vectors  
- **SimpleRNN Layer:** 64 recurrent units with dropout  
- **Dense Output:** Sigmoid activation for binary classification  

---

## 📂 Project Structure

\`\`\`
simple-rnn-imdb-sentiment/
├── app.py                   # Streamlit app
├── simple_rnn_imdb.h5       # Trained model
├── requirements.txt         # Python dependencies
├── simple_rnn_project.ipynb # Model training notebook
└── README.md                # This file
\`\`\`

---

## 🚀 How to Run Locally

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/Mukunda44/simple-rnn-imdb-sentiment.git
   cd simple-rnn-imdb-sentiment
   \`\`\`

2. **(Optional) Create and activate a virtual environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\\Scripts\\activate     # For Windows
   \`\`\`

3. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Run the Streamlit app**
   \`\`\`bash
   streamlit run app.py
   \`\`\`

---

## 📦 Dependencies

All required packages are in \`requirements.txt\`. Core libraries:

- \`tensorflow\`
- \`streamlit\`
- \`numpy\`

---

## 📈 Model Training

If you want to retrain the model from scratch, refer to:
- \`simple_rnn_project.ipynb\` — includes full training, preprocessing, and model saving code.

---

## 🖼️ App Preview

<img width="1162" height="598" alt="image" src="https://github.com/user-attachments/assets/d0fd679e-f510-4833-8d5e-54fd959e9601" />


---

## ✅ Future Improvements

- Replace \`SimpleRNN\` with \`LSTM\` or \`GRU\`  
- Use custom tokenizer instead of Keras default word index  
- Add word cloud or review visualization  
- Deploy to Streamlit Cloud or Hugging Face Spaces  

---

## 📝 License

MIT License  
Free to use for learning and educational purposes.

---

## 🙌 Acknowledgments

- IMDB dataset via [Keras Datasets](https://keras.io/api/datasets/imdb/)
- TensorFlow/Keras for model development  
- Streamlit for interactive UI

---
EOF
