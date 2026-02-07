# SRT File Translator 🎬

A powerful subtitle file translator with a user-friendly Gradio interface. Supports multiple translation services including OpenAI-compatible APIs, LibreTranslate, and Google Translate.

## Features ✨

- **Multiple Translation Services**:
  - OpenAI-compatible APIs (OpenAI, Ollama, LM Studio, etc.)
  - LibreTranslate (self-hosted or public)
  - Google Translate API

- **User-Friendly Interface**:
  - Drag-and-drop or file browser for SRT files
  - Edit source subtitles before translation
  - Edit translated subtitles after translation
  - Real-time progress logging
  - Download translated files

- **Flexible Translation**:
  - Translate entire file at once
  - Translate individual lines
  - Support for 20+ languages
  - No timeout between requests for better service compatibility

- **Professional Logging**:
  - Pretty-printed logs with timestamps
  - Verbose mode for detailed debugging
  - Step-by-step progress tracking

## Installation 🚀

### Local Installation

1. **Clone or download the files**:
   ```bash
   # Make sure you have these files:
   # - srt_translator.py
   # - requirements.txt
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python srt_translator.py
   ```

4. **Open your browser**:
   - Navigate to `http://localhost:7860`

### Docker Installation

1. **Build the Docker image**:
   ```bash
   docker build -t srt-translator .
   ```

2. **Run the container**:
   ```bash
   docker run -p 7860:7860 -v $(pwd)/data:/app/data srt-translator
   ```

3. **Or use Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - Navigate to `http://localhost:7860`

## Usage Guide 📖

### Step 1: Configure Translation Service

**Left Column - Service Configuration**

Choose your preferred translation service:

#### Option A: OpenAI-Compatible API
- **Service Type**: Select "OpenAI-Compatible API"
- **API Host**: Enter your API endpoint
  - OpenAI: `https://api.openai.com`
  - Ollama: `http://localhost:11434`
  - LM Studio: `http://localhost:1234`
- **API Key**: Your API key (use "ollama" for Ollama if no key required)
- **Model**: Model name (e.g., `gpt-4`, `llama2`, `mistral`)

#### Option B: LibreTranslate
- **Service Type**: Select "LibreTranslate"
- **Host**: LibreTranslate server URL
  - Public: `https://libretranslate.com`
  - Self-hosted: `http://localhost:5000`
- **API Key**: Optional (required for some instances)

#### Option C: Google Translate
- **Service Type**: Select "Google Translate"
- **API Key**: Your Google Cloud API key

**Toggle Verbose Logging** for detailed process information.

### Step 2: Load Source File

**Middle Column - Source Configuration**

1. **Select Source Language**: Choose the language of your SRT file
2. **Upload SRT File**: Drag-and-drop or click to browse
3. **Click "Load File"**: Parses and displays the subtitle entries
4. **Edit if needed**: Modify any source text in the editable table

### Step 3: Configure Target Language

**Right Column - Translation Configuration**

1. **Select Target Language**: Choose your desired translation language
2. **Click "Translate All"**: Translates all subtitle entries
3. **Review & Edit**: Check translations and make any necessary edits
4. **Click "Download Translation"**: Save your translated SRT file

### Step 4: Monitor Progress

**Bottom Section - Logs**

- Watch real-time progress with pretty-printed logs
- Track which entry is being processed
- See success/error messages
- Enable verbose mode for detailed information

## File Management 📁

All temporary and permanent files are saved in the same directory where the script is executed:

- **Temporary source files**: `temp_source_YYYYMMDD_HHMMSS.srt`
- **Temporary target files**: `temp_target_YYYYMMDD_HHMMSS.srt`
- **Final translation**: `translated_YYYYMMDD_HHMMSS.srt`

## Language Codes 🌍

Common language codes supported:

| Code | Language   | Code | Language    |
|------|------------|------|-------------|
| en   | English    | es   | Spanish     |
| fr   | French     | de   | German      |
| it   | Italian    | pt   | Portuguese  |
| ru   | Russian    | ja   | Japanese    |
| ko   | Korean     | zh   | Chinese     |
| ar   | Arabic     | hi   | Hindi       |
| nl   | Dutch      | pl   | Polish      |
| tr   | Turkish    | sv   | Swedish     |

## SRT File Format 📄

The script handles standard SRT format:

```
1
00:00:01,000 --> 00:00:04,000
This is the first subtitle

2
00:00:05,000 --> 00:00:08,000
This is the second subtitle
```

## Troubleshooting 🔧

### Common Issues

1. **"Failed to load SRT file"**
   - Ensure file is in valid SRT format
   - Check file encoding (should be UTF-8)

2. **"Translation failed"**
   - Verify API credentials are correct
   - Check network connectivity
   - Ensure service endpoint is accessible

3. **"Service requires host/API key"**
   - Fill in all required configuration fields
   - API keys should not have leading/trailing spaces

4. **Slow translations**
   - Normal for large files (no timeout between requests)
   - Consider using batch processing for very large files

### Docker Issues

1. **Port already in use**:
   ```bash
   # Use different port
   docker run -p 8080:7860 -v $(pwd)/data:/app/data srt-translator
   ```

2. **Permission issues with volumes**:
   ```bash
   # Create directories first
   mkdir -p data temp
   chmod 777 data temp
   ```

## API Examples 🔌

### Using with Ollama (Local)

```
Service Type: OpenAI-Compatible API
API Host: http://localhost:11434/v1
API Key: ollama
Model: llama2
```

### Using with LM Studio (Local)

```
Service Type: OpenAI-Compatible API
API Host: http://localhost:1234/v1
API Key: lm-studio
Model: your-model-name
```

### Using with LibreTranslate (Public)

```
Service Type: LibreTranslate
Host: https://libretranslate.com
API Key: (leave empty for public instance)
```

### Using with OpenAI

```
Service Type: OpenAI-Compatible API
API Host: https://api.openai.com/v1
API Key: sk-your-api-key-here
Model: gpt-4-turbo-preview
```

## Advanced Features 🎯

### Individual Line Translation

1. Load your SRT file
2. Edit the specific line in the source editor
3. Use the translate button for that entry
4. Review and edit the translation

### Batch Processing

The script processes entries sequentially without timeouts, making it suitable for:
- Large subtitle files
- Rate-limited APIs
- Free-tier services with slower response times

### Custom Service Integration

The OpenAI-compatible API option works with any service that implements the OpenAI chat completion format:
- Local LLMs (Ollama, LM Studio, text-generation-webui)
- Cloud services (OpenAI, Anthropic with adapter, etc.)
- Custom endpoints

## Contributing 🤝

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License 📝

This project is open source and available for personal and commercial use.

## Support 💬

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in verbose mode
3. Ensure your translation service is properly configured

---

**Happy Translating! 🎉**
