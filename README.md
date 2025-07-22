# 🇬🇧 English Sentiment Analysis with Fine-tuned BERT

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange)
![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Metrics-red)
![Docker](https://img.shields.io/badge/Docker-Container-blue)

---

##  Project Overview

This **Natural Language Processing (NLP)** project focuses on **fine-tuning a pre-trained BERT model for sentiment analysis** on English product reviews. The primary goal is to accurately classify the sentiment of these reviews (negative, neutral, positive) to help businesses better understand customer feedback and improve their products or services.

As a freelancer specializing in Machine Learning and NLP, I developed this solution to demonstrate my ability to build and deploy state-of-the-art AI models for practical applications, highlighting skills in model fine-tuning, API development, and project modularity. This project is designed for both **local execution** and **cloud deployment (e.g., via Google Colab and Docker)**.

---

##  Key Features

* **Model**: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) (A standard BERT model pre-trained on a vast English corpus).
* **Dataset**: A subset of 10,000 reviews from the [`amazon_reviews_multi` (English)](https://huggingface.co/datasets/amazon_reviews_multi) dataset.
* **Task**: Multi-class classification (negative, neutral, positive sentiment).
* **Libraries**: Hugging Face `transformers` and `datasets`, `scikit-learn` for metrics, `FastAPI` for the inference API.
* **Environment**: Developed and tested on Google Colab with T4 GPU, deployable locally or via Docker.
* **API**: A RESTful API built with FastAPI to serve sentiment predictions.

---

##  Project Structure

The repository is organized for clarity and ease of use:

english-sentiment-analysis/
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
    git clone [https://github.com/your-username/english-sentiment-analysis.git](https://github.com/your-username/english-sentiment-analysis.git)
    cd english-sentiment-analysis
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
    Execute the training script. This will download the dataset, preprocess it, fine-tune the **English BERT model**, and save the best model to `models/final_bert_sentiment_model`. This step requires a GPU for efficient training.
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
      "text": "This product is absolutely amazing, I love it!"
    }
    ```
    And receive a sentiment prediction.

### Google Colab (All-in-One Notebook)

You can run the entire project, including training and a publicly exposed API (via ngrok), directly in a Google Colab notebook.

1.  **Open a new Colab notebook**: Go to [colab.research.google.com](https://colab.research.google.com/).
2.  **Configure GPU**:
    * Go to **"Runtime" > "Change runtime type"**.
    * Select **"T4 GPU"** (or another available GPU) under "Hardware accelerator". Click **"Save"**.
3.  **Install Libraries and Create Folders**:
    In a new code cell, paste and run:
    ```python
    !mkdir -p src models notebooks
    !pip install -q transformers datasets sklearn evaluate torch fastapi uvicorn pydantic accelerate pyngrok
    print("Environment setup complete!")
    ```
4.  **Create `src/train.py` (updated for English)**:
    In the next cell, paste the updated content for `src/train.py` (as provided in the previous instructions, with English dataset and BERT checkpoint changes) prefixed with `%%writefile src/train.py`.
5.  **Create `src/app.py` (updated for English)**:
    In the next cell, paste the updated content for `src/app.py` (as provided in the previous instructions, with English labels and API description changes) prefixed with `%%writefile src/app.py`.
6.  **Run Training (`src/train.py`)**:
    In the next cell, run:
    ```bash
    !python src/train.py
    ```
    This will train and save the **English sentiment model**.
7.  **Start API with ngrok**:
    In the next cell, paste and run the `pyngrok` code. **Crucially, replace `"YOUR_AUTHTOKEN"` with your actual ngrok authentication token.**
    ```python
    # Ensure src.app is imported correctly if running in separate cells
    from src.app import app as fastapi_app # Import the FastAPI app

    # Function to start Uvicorn in a separate thread
    def run_uvicorn():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

    # Ngrok authentication (replace YOUR_AUTHTOKEN)
    NGROK_AUTH_TOKEN = "YOUR_AUTHTOKEN"
    if NGROK_AUTH_TOKEN == "YOUR_AUTHTOKEN":
        print("WARNING: Replace 'YOUR_AUTHTOKEN' with your ngrok token from [https://dashboard.ngrok.com/auth/your-authtoken](https://dashboard.ngrok.com/auth/your-authtoken).")
    else:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    # Start API & ngrok tunnel
    print("Starting FastAPI API in background...")
    api_thread = Thread(target=run_uvicorn)
    api_thread.start()
    time.sleep(5)
    public_url = ngrok.connect(8000).public_url
    print(f"Your API is publicly available at: {public_url}")
    print(f"Swagger UI: {public_url}/docs")
    ```
    Colab will print a public URL (e.g., `https://<random_id>.ngrok-free.app`). This is your live API endpoint.
8.  **Test the API (from Colab)**:
    In the next cell, paste and run the testing code. **Replace `"YOUR_NGROK_URL_HERE"` with the URL obtained from ngrok.**
    ```python
    import requests
    import json

    public_url = "YOUR_NGROK_URL_HERE" # <--- PASTE YOUR NGROK URL HERE

    if public_url == "YOUR_NGROK_URL_HERE":
        print("WARNING: You must copy and paste the public ngrok URL from the cell above!")
    else:
        api_endpoint = f"{public_url}/predict_sentiment"
        headers = {"Content-Type": "application/json"}
        test_text = "This is a great product!"
        data = {"text": test_text}
        try:
            response = requests.post(api_endpoint, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            print(f"\nText: '{result['text']}'")
            print(f"Predicted Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence_scores']}")
        except requests.exceptions.RequestException as e:
            print(f"\nError during API request: {e}")
    ```

---

##  Results and Performance

After 3 training epochs, the fine-tuned model achieved the following performance on the test set (using a 10,000-sample subset of the English Amazon reviews dataset):

| Metric   | Value    |
| :------- | :------- |
| Accuracy | XX.XX%   |
| F1-Macro | XX.XX%   |

*(**Note**: Please replace `XX.XX%` with your actual results after running the training script.)*

These results demonstrate that the fine-tuned BERT model is capable of accurately classifying the sentiment of English reviews, providing a solid foundation for real-world applications.

---

##  Dockerization (Recommended for Production)

For easy deployment and reproducibility, the API can be containerized using Docker.

1.  **Ensure model is trained**: Make sure `models/final_bert_sentiment_model` exists from the training step.
2.  **Build the Docker image**: From the root of your project directory (where `Dockerfile` is located):
    ```bash
    docker build -t english-sentiment-api .
    ```
3.  **Run the Docker container**:
    ```bash
    docker run -p 8000:8000 english-sentiment-api
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

* **Email**: [your.email@example.com](mailto:your.email@example.com)
* **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/your-linkedin-profile/)
* **Portfolio**: [Your Personal Portfolio (if any)](https://www.your-portfolio.com)

---

##  License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

