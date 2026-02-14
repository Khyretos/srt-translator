# ⚠️ **AI ALERT**

As a software developer with limited Python expertise, I directed the development of srt-translator by iteratively prompting AI language models (Claude and DeepSeek) to generate the majority of the code. Through systematic debugging, precise requirements, and continuous validation of the AI's output, I guided the project from concept to a fully functional application. This process demonstrates my ability to leverage AI tools effectively while retaining full ownership of problem‑solving and architectural decisions.

# 🎬 SRT File Translator

A powerful and user‑friendly subtitle translation tool with a **Gradio** web interface.  
Translate your `.srt` subtitle files using **OpenAI‑compatible APIs**, **LibreTranslate**, or **Google Translate**.  
Edit subtitles directly in the table, resume interrupted translations, and monitor progress with detailed logs.

---

## ✨ Features

- **Multiple Translation Services**
  - OpenAI‑compatible APIs (e.g., OpenAI, LocalAI, Ollama, vLLM)
  - LibreTranslate (self‑hosted or public instances)
  - Google Translate (official API)
- **Interactive Web UI** built with Gradio – upload, edit, translate, and download in one place
- **Editable Subtitle Table** – modify source text or timecodes; sequences renumber automatically
- **Resume Interrupted Translation** – continue from where you left off
- **Streaming Logs** – real‑time progress and detailed error messages
- **Verbose Logging** option for debugging
- **Pause / Stop** translation at any time
- **Automatic Temporary Backup** of source and translated files
- **Docker Support** – easy deployment with Docker Compose
- **Lightweight** – only requires Python and a few libraries

---

## 📦 Requirements

- Python 3.8 or higher
- Internet connection (for external translation APIs)
- (Optional) Docker & Docker Compose

---

## 🚀 Installation

### Local Installation

1. **Clone the repository** (or download the files):

   ```bash
   git clone https://github.com/yourusername/srt-translator.git
   cd srt-translator
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python srt_translator.py
   ```
   The Gradio interface will be available at `http://localhost:7860`.

### Docker Deployment

1. **Ensure Docker and Docker Compose are installed.**
2. **In the project directory, run:**

   ```bash
   docker-compose up -d
   ```

3. Access the UI at `http://localhost:7860`.

The Docker setup mounts `./data` and `./temp` folders to persist uploaded files and temporary data.

## 🖥️ Usage Guide

### 1. Select Translation Service

- **OpenAI‑Compatible API**
  - **Host:** e.g., `http://localhost:11434/v1` (for Ollama) or `https://api.openai.com/v1`
  - **API Key:** Your API key (if required)
  - **Model:** e.g., `gpt-4`, `llama2`, `mistral`

* **LibreTranslate**
  - Host: e.g., https://libretranslate.com or your self‑hosted instance
  - API Key: Optional (if your instance requires it)

* Google Translate
  - Google API Key: Obtain from Google Cloud Console

### 2. Load an SRT File

Click “Upload SRT File” and select your `.srt` file.
Press “Load File” – the table will populate with subtitle entries.

### 3. Edit Subtitles (Optional)

You can modify the Text column directly.
Add or remove rows by right‑clicking in the table.
Sequence numbers update automatically when you change the table.

### 4. Translate

- **Translate All** – starts translating from the first entry (or from the last stopped position).
- **Stop** – pauses the translation; you can later Continue from Stop.
- **Translate Single** – enter a sequence number and click to translate only that entry.

The logs panel shows real‑time progress and any errors.

### 5. Download Translated File

After translation, click **“Download Translation”** to save the `.srt` file with the translated text.

## ⚙️ Configuration

- **Verbose Logging** – enables detailed debug output (useful for troubleshooting).
- The application stores temporary files in the `temp` folder (or `./temp` in Docker).
- Language codes follow [ISO 639‑1](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) (e.g., `en`, `es`, `fr`).

## 🔧 How It Works

1. **Parsing** – The .srt file is read and split into individual entries (sequence number, timecode, text).
2. **Translation** – For each entry, the text is sent to the chosen translation service.
3. **Retry Logic** – Each translation request is retried up to 5 times with a 2‑second delay on failure.
4. **Progress Tracking** – The current index is stored, allowing you to resume after a stop.
5. **Saving** – The translated subtitles are written back in the same SRT format.

## 🌐 Supported Translation Services

| Service           | Requirements                     | Notes                                                                           |
| ----------------- | -------------------------------- | ------------------------------------------------------------------------------- |
| OpenAI‑Compatible | Host, API Key (if needed), Model | Works with any OpenAI‑compatible endpoint (OpenAI, LocalAI, Ollama, vLLM, etc.) |
| LibreTranslate    | Host, optional API Key           | Can be self‑hosted for privacy                                                  |
| Google Translate  | Google API Key                   | Official Google Cloud Translation API                                           |

## 🐞 Troubleshooting

- **No translation service selected** – Ensure you’ve filled in the required fields for your chosen service.
- **Connection errors** – Verify the host URL and network connectivity.
- **Rate limiting / timeouts** – The built‑in retry logic will attempt again; you can also adjust the code for longer timeouts.
- **Table not updating after translation** – The UI refreshes automatically; if not, try clicking elsewhere.
- **Docker issues** – Check that ports are not in use and volumes are writable.

```text
📁 Project Structure
text
srt-translator/
├── srt_translator.py # Main application
├── requirements.txt # Python dependencies
├── docker-compose.yml # Docker Compose configuration
├── Dockerfile # Docker image definition (implied but not shown; can be created)
├── data/ # Mounted volume for persistent data (optional)
├── temp/ # Temporary files storage
└── README.md # This file
```

## 📝 License

This project is open‑source and available under the `MIT License`. Feel free to use, modify, and distribute it.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

## 📧 Contact

For questions or support, please open an issue on the GitHub repository (replace with your actual URL).

Enjoy translating your subtitles! 🎉
