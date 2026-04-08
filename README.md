# 🤖 AI Assistant

An intelligent AI Assistant built to automate tasks, answer queries, and assist users using modern AI capabilities.

## 🚀 Features

* 💬 Natural language conversation
* ⚡ Fast API using FastAPI
* ☁️ Deployable on Google Cloud Run
* 🔗 Modular agent architecture
* 🧠 AI-powered responses
* 📦 Easy to extend tools

## 🛠️ Tech Stack

* Python
* FastAPI
* Google ADK
* Cloud Run
* Uvicorn
* Pydantic
<img width="1906" height="867" alt="image" src="https://github.com/user-attachments/assets/092fc73c-e8fd-497c-800e-408842a8cb38" />

## 📂 Project Structure

```
.
├── main.py
├── requirements.txt
├── agent/
├── tools/
├── .env
└── README.md
```


## ▶️ Run Locally

```bash
uvicorn main:app --reload
```

## ☁️ Deploy to Cloud Run

```bash
uvx --from google-adk==1.14.0 \
adk deploy cloud_run \
  --project=$PROJECT_ID \
  --region=us-central1 \
  --service_name=ai-assistant \
  --with_ui \
  . \
  -- \
  --service-account=$SERVICE_ACCOUNT
```

## 🔐 Environment Variables

Create `.env` file:

```
PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path-to-key.json
```

## 📌 Usage

* Start server
* Open provided URL
* Chat with AI assistant

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

## 📄 License

MIT License

---

⭐ If you like this project, give it a star!
