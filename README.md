# 📰 Fake News Detection using BERT

A machine learning project that detects whether a news statement is **Real** or **Fake** using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model.

The system is trained on the **LIAR dataset** and deployed with a **Streamlit web interface**, allowing users to input a statement and instantly receive a prediction with a confidence score.

---

# 🚀 Project Overview

Fake news and misinformation spread rapidly across digital platforms.
This project demonstrates how **Natural Language Processing (NLP)** and **Transformer-based models** can be used to detect misinformation automatically.

The system performs the following steps:

1. User enters a news statement
2. The text is tokenized using a BERT tokenizer
3. The fine-tuned BERT model predicts whether the statement is **Real or Fake**
4. The result is displayed with a confidence score through a **Streamlit web app**

---

# 📊 Dataset

This project uses the **LIAR dataset**, a widely used benchmark dataset for political fact-checking.

Dataset details:

* ~12,800 labeled statements
* Labels include:

  * true
  * mostly-true
  * half-true
  * barely-true
  * false
  * pants-fire

For this project, labels were simplified into **binary classes**:

| Original Label | Converted Label |
| -------------- | --------------- |
| true           | Real            |
| mostly-true    | Real            |
| half-true      | Real            |
| barely-true    | Fake            |
| false          | Fake            |
| pants-fire     | Fake            |

---

# 🧠 Model Architecture

The project uses **BERT Base (bert-base-uncased)** for text classification.

Pipeline:

Text Input
↓
BERT Tokenizer
↓
BERT Encoder
↓
Classification Layer
↓
Prediction (Real / Fake)

Training configuration:

* Model: `bert-base-uncased`
* Maximum sequence length: **128 tokens**
* Batch size: **8**
* Epochs: **2–3**
* Learning rate: **2e-5**
* Framework: **PyTorch + HuggingFace Transformers**

---

# 🖥️ Web Application

A **Streamlit app** is included so users can interact with the trained model.

Features:

* Enter any statement
* Real-time prediction
* Confidence score display
* Simple interactive UI

Example output:

Real News ✅ (0.88 confidence)

or

Fake News ❌ (0.76 confidence)

---

# 📂 Project Structure

```
fake-news-detector
│
├── dataset/
│   ├── train.tsv
│   ├── test.tsv
│   └── valid.tsv
│
├── model/            # Trained model (not included in repo due to size)
│
├── app.py            # Streamlit web app
├── train.py          # Model training script
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

Create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit app:

```
streamlit run app.py
```

Open the browser:

```
http://localhost:8501
```

---

# 🧪 Example Test Inputs

### Real News

```
India successfully landed the Chandrayaan-3 spacecraft near the Moon's south pole.
```

### Fake News

```
NASA confirmed that Earth will experience six days of darkness next month.
```

---

# 📈 Future Improvements

Possible improvements for this project:

* Train on larger fake news datasets
* Multi-class fact-checking classification
* Explainable AI for prediction reasoning
* Fact verification using external sources
* Real-time misinformation monitoring

---

# 🛠 Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* Streamlit
* Pandas
* Scikit-learn
* Datasets library

---

# 🎓 Learning Outcomes

This project demonstrates:

* Transformer model fine-tuning
* NLP text classification
* Dataset preprocessing
* ML model deployment
* Building interactive ML applications

---

# 👨‍💻 Author

**Kartik Kumar**
B.Tech Computer Science (2023-2027)

Interested in **Machine Learning, AI systems, and data-driven applications**.
