# 🇮🇹 Italian Sentiment Analysis with Fine-tuned BERT

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange)
![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Metrics-red)
![Docker](https://img.shields.io/badge/Docker-Container-blue)

---

##  Project Overview

This **Natural Language Processing (NLP)** project focuses on **fine-tuning a pre-trained BERT model for sentiment analysis** on Italian product reviews. The goal is to classify the sentiment of reviews (negative, neutral, positive) to help businesses better understand customer feedback and improve their products or services.

As a freelancer specializing in Machine Learning and NLP, I developed this solution to demonstrate my ability to build and deploy state-of-the-art AI models for practical applications, highlighting skills in model fine-tuning, API development, and project modularity.

---

##  Key Features

* **Model**: [`dbmdz/bert-base-italian-uncased`](https://huggingface.co/dbmdz/bert-base-italian-uncased) (A BERT model pre-trained specifically for Italian).
* **Dataset**: A subset of 10,000 reviews from the [`amazon_reviews_multi` (Italian)](https://huggingface.co/datasets/amazon_reviews_multi) dataset.
* **Task**: Multi-class classification (negative, neutral, positive sentiment).
* **Libraries**: Hugging Face `transformers` and `datasets`, `scikit-learn` for metrics, `FastAPI` for the inference API.
* **Environment**: Developed and tested on Google Colab with T4 GPU, deployable locally or via Docker.
* **API**: A RESTful API built with FastAPI to serve sentiment predictions.

---

##  Project Structure

The repository is organized for clarity and ease of use:

bert-sentiment-italian/
├── notebooks/
│   └── train_model.ipynb   # Google Colab notebook for step-by-step training.
├── src/
│   ├── train.py            # Python script for model training and saving.
│   └── app.py              # FastAPI application script for inference.
├── models/
│   └── final_bert_sentiment_model/ # Directory where the trained BERT model and tokenizer are saved.
├── requirements.txt        # List of Python dependencies for the project.
├── Dockerfile              # Docker configuration for containerizing the API.
└── README.md               # This README file.


---

##  How to Run the Project

You can run this project locally or entirely within Google Colab.

### Local Setup (Recommended for Full Control)

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/bert-sentiment-italian.git](https://github.com/your-username/bert-sentiment-italian.git)
    cd bert-sentiment-italian
    ```
2.  **Create necessary folders**:
    ```bash
    mkdir -p src models notebooks
    ```
    Then, place `train.py` and `app.py` into the `src` folder.
3.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```
4.  **Install dependencies**:
    Ensure your `requirements.txt` file (in the root directory) contains all the libraries listed above.
    ```bash
    pip install -r requirements.txt
    ```
5.  **Train the Model**:
    Execute the training script. This will download the dataset, preprocess it, fine-tune the BERT model, and save the best model to `models/final_bert_sentiment_model`. This step requires a GPU for efficient training.
    ```bash
    python src/train.py
    ```
    *(If you don't have a local GPU, perform the training step on Google Colab (see section below) and then download the `models/final_bert_sentiment_model` folder to your local setup.)*

6.  **Start the FastAPI API**:
    Once the model is saved, launch the API:
    ```bash
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
    ```
    The `--reload` flag is useful for development; remove it for production deployments.

7.  **Test the API**:
    Open your web browser and navigate to:
    * `http://127.0.0.1:8000/` for a welcome message.
    * `http://127.0.0.1:8000/docs` for the interactive Swagger UI documentation, where you can test the `/predict_sentiment` endpoint.
    * `http://127.0.0.1:8000/redoc` for alternative documentation.

    You can send a POST request to `http://127.0.0.1:8000/predict_sentiment` with a JSON body like:
    ```json
    {
      "text": "Questo prodotto è fantastico, lo adoro!"
    }
    ```
    And receive a sentiment prediction.

### Google Colab (All-in-One Notebook)

You can run the entire project, including training and a publicly exposed API (via ngrok), directly in a Google Colab notebook.

1.  **Open a new Colab notebook**: Go to [colab.research.google.com](https://colab.research.google.com/).
2.  **Configure GPU**:
    * Go to **"Runtime" > "Change runtime type"**.
    * Select **"T4 GPU"** (or another available GPU) under "Hardware accelerator". Click **"Save"**.
3.  **Install Libraries**:
    In a new code cell, paste and run:
    ```python
    !pip install -q transformers datasets sklearn evaluate torch fastapi uvicorn pydantic accelerate
    print("Libraries installed successfully!")
    ```
4.  **Run Training and Model Saving**:
    In the next code cell, paste the full content of `src/train.py` (as provided in the code examples). Execute this cell. This will train your model and save it to `models/final_bert_sentiment_model` within your Colab environment.
5.  **Create `app.py` for FastAPI**:
    In the next code cell, paste the full content of `src/app.py` (as provided in the code examples) prefixed with `%%writefile src/app.py`:
    ```python
    %%writefile src/app.py
    # (Paste the full content of src/app.py here)
    ```
    Execute this cell to create the `app.py` file.
6.  **Start API with ngrok**:
    In the next code cell, paste and run the following. **Crucially, replace `"YOUR_AUTHTOKEN"` with your actual ngrok authentication token.**
    * **Get your ngrok Auth Token**: Go to [https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup) to sign up for a free account, then find your token at [https://dashboard.ngrok.com/auth/your-authtoken](https://dashboard.ngrok.com/auth/your-authtoken).
    ```python
    # Install ngrok to expose the API port
    !pip install -q pyngrok

    from pyngrok import ngrok
    import os
    import time
    from threading import Thread

    # Import the FastAPI app from the created file
    from src.app import app as fastapi_app

    # Function to start Uvicorn in a separate thread
    def run_uvicorn():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

    # Ngrok authentication
    NGROK_AUTH_TOKEN = "YOUR_AUTHTOKEN" # <--- REPLACE THIS WITH YOUR ACTUAL NGROK TOKEN

    if NGROK_AUTH_TOKEN == "YOUR_AUTHTOKEN":
        print("WARNING: You must replace 'YOUR_AUTHTOKEN' with your ngrok authentication token!")
        print("Go to [https://dashboard.ngrok.com/auth/your-authtoken](https://dashboard.ngrok.com/auth/your-authtoken) to get it.")
        print("Without the token, ngrok will not work.")
    else:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        print("Ngrok token configured.")

    # Start the API in a separate thread
    print("Starting FastAPI API in the background...")
    api_thread = Thread(target=run_uvicorn)
    api_thread.start()
    time.sleep(5) # Give the API time to start

    # Start the ngrok tunnel
    print("Starting ngrok tunnel...")
    public_url = ngrok.connect(8000).public_url
    print(f"Your API is publicly available at: {public_url}")
    print(f"Swagger UI documentation: {public_url}/docs")
    print(f"Redoc UI: {public_url}/redoc")
    print("\nNote: This link will be valid as long as the Colab session is active.")
    ```
    Colab will print a public URL (e.g., `https://<random_id>.ngrok-free.app`). This is your live API endpoint.

7.  **Test the API (from Colab)**:
    In the next cell, paste and run the following. **Replace `"YOUR_NGROK_URL_HERE"` with the URL you obtained from the previous step.**
    ```python
    import requests
    import json

    public_url = "YOUR_NGROK_URL_HERE" # <--- PASTE YOUR NGROK URL HERE

    if public_url == "YOUR_NGROK_URL_HERE":
        print("WARNING: You must copy and paste the public ngrok URL from the previous cell here!")
    else:
        api_endpoint = f"{public_url}/predict_sentiment"
        headers = {"Content-Type": "application/json"}

        test_texts = [
            "Questo prodotto è fantastico, lo adoro!",
            "La qualità non è delle migliori, sono un po' deluso.",
            "Il prodotto è arrivato in tempo, ma non è nulla di eccezionale."
        ]

        print(f"\nTesting the API at {api_endpoint}...")
        for text in test_texts:
            data = {"text": text}
            try:
                response = requests.post(api_endpoint, headers=headers, data=json.dumps(data))
                response.raise_for_status()
                result = response.json()
                print(f"\nText: '{result['text']}'")
                print(f"Predicted Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence_scores']}")
            except requests.exceptions.RequestException as e:
                print(f"\nError during API request for text '{text}': {e}")
    ```

---

##  Results and Performance

After 3 training epochs, the fine-tuned model achieved the following performance on the test set (using a 10,000-sample subset of the Amazon reviews dataset):

| Metric   | Value    |
| :------- | :------- |
| Accuracy | XX.XX%   |
| F1-Macro | XX.XX%   |

*(**Note**: Please replace `XX.XX%` with your actual results after running the training script.)*

These results demonstrate that the fine-tuned BERT model is capable of accurately classifying the sentiment of Italian reviews, providing a solid foundation for real-world applications.

---

##  Dockerization (Recommended for Production)

For easy deployment and reproducibility, the API can be containerized using Docker.

1.  **Ensure model is trained**: Make sure `models/final_bert_sentiment_model` exists from the training step.
2.  **Build the Docker image**: From the root of your project directory (where `Dockerfile` is located):
    ```bash
    docker build -t sentiment-italian-api .
    ```
3.  **Run the Docker container**:
    ```bash
    docker run -p 8000:8000 sentiment-italian-api
    ```
    Your API will now be accessible at `http://localhost:8000`.

---

##  Future Enhancements

* **Expanded Dataset**: Train on the full `amazon_reviews_multi` dataset for improved generalization.
* **More Advanced Fine-tuning**: Explore techniques like learning rate scheduling, different optimizers, or more epochs.
* **Error Analysis**: Implement a module to analyze misclassified examples to identify model weaknesses.
* **Deployment to Cloud Platforms**: Deploy the Dockerized API to platforms like Google Cloud Run, AWS ECS, or Hugging Face Spaces for scalable inference.
* **Streamlit/Gradio UI**: Develop a simple web interface for easy interaction with the model without needing `curl` or `requests`.
* **Model Monitoring**: Implement tools to monitor model performance and data drift in production.

---

##  Contact

I'm available for collaborations and freelance projects related to Machine Learning and NLP. Feel free to reach out:

* **Email**: (gliracurcio@gmail.com)


---

##  License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
