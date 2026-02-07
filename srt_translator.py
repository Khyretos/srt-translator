#!/usr/bin/env python3
"""
SRT File Translator with Gradio UI
Supports OpenAI-compatible APIs, LibreTranslate, and Google Translate
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import requests


class Logger:
    """Pretty logger for translation process"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logs = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp and level"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
        return log_entry

    def debug(self, message: str):
        """Log debug message only if verbose is enabled"""
        if self.verbose:
            return self.log(message, "DEBUG")
        return None

    def info(self, message: str):
        """Log info message"""
        return self.log(message, "INFO")

    def warning(self, message: str):
        """Log warning message"""
        return self.log(message, "WARNING")

    def error(self, message: str):
        """Log error message"""
        return self.log(message, "ERROR")

    def get_logs(self) -> str:
        """Get all logs as a formatted string"""
        return "\n".join(self.logs)

    def clear(self):
        """Clear all logs"""
        self.logs = []


class SRTEntry:
    """Represents a single SRT subtitle entry"""

    def __init__(self, sequence: int, timecode: str, text: str):
        self.sequence = sequence
        self.timecode = timecode
        self.text = text
        self.translated_text = ""

    def __str__(self):
        return f"{self.sequence}\n{self.timecode}\n{self.text}\n"

    def to_dict(self):
        """Convert to dictionary for UI display"""
        return {
            "sequence": self.sequence,
            "timecode": self.timecode,
            "text": self.text,
            "translated_text": self.translated_text,
        }


class SRTParser:
    """Parse and write SRT files"""

    @staticmethod
    def parse(content: str, logger: Logger) -> List[SRTEntry]:
        """Parse SRT file content into entries"""
        logger.info("📖 Parsing SRT file...")
        entries = []

        # Split by double newlines to get individual entries
        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            lines = [
                line.strip("\ufeff") for line in lines if line.strip("\ufeff") != ""
            ]
            if len(lines) < 3:
                continue

            try:
                sequence = int(lines[0].strip())
                timecode = lines[1].strip()
                text = "\n".join(lines[2:])

                entries.append(SRTEntry(sequence, timecode, text))
                logger.debug(f"   Parsed entry {sequence}: {timecode}")
            except (ValueError, IndexError) as e:
                logger.warning(f"   ⚠️  Skipped malformed entry: {e}")
                continue

        logger.info(f"✅ Parsed {len(entries)} subtitle entries")
        return entries

    @staticmethod
    def write(
        entries: List[SRTEntry],
        filepath: str,
        logger: Logger,
        use_translated: bool = False,
    ):
        """Write entries to SRT file"""
        logger.info(f"💾 Writing SRT file: {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            for entry in entries:
                text = (
                    entry.translated_text
                    if use_translated and entry.translated_text
                    else entry.text
                )
                f.write(f"{entry.sequence}\n")
                f.write(f"{entry.timecode}\n")
                f.write(f"{text}\n\n")

        logger.info(f"✅ File saved successfully")


class TranslationService:
    """Base class for translation services"""

    def __init__(self, logger: Logger):
        self.logger = logger

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source to target language"""
        raise NotImplementedError


class OpenAITranslationService(TranslationService):
    """OpenAI-compatible API translation service"""

    def __init__(self, host: str, api_key: str, model: str, logger: Logger):
        super().__init__(logger)
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.model = model

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using OpenAI-compatible API"""
        self.logger.debug(f"   🤖 Translating with AI: {self.model}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        prompt = f"You are a professional translator. Translate from {source_lang} to {target_lang}. Preserve the meaning and tone and only return the translated text, nothing else.\n\nText: {text}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = requests.post(
                f"{self.host}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            translated = result["choices"][0]["message"]["content"].strip()
            self.logger.debug(f"   ✓ Translation received")
            return translated

        except Exception as e:
            self.logger.error(f"   ❌ AI translation failed: {e}")
            raise


class LibreTranslateService(TranslationService):
    """LibreTranslate API service"""

    def __init__(self, host: str, api_key: str, logger: Logger):
        super().__init__(logger)
        self.host = host.rstrip("/")
        self.api_key = api_key

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using LibreTranslate"""
        self.logger.debug(f"   🌐 Translating with LibreTranslate")

        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text",
        }

        if self.api_key:
            payload["api_key"] = self.api_key

        try:
            response = requests.post(f"{self.host}/translate", json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            translated = result["translatedText"]
            self.logger.debug(f"   ✓ Translation received")
            return translated

        except Exception as e:
            self.logger.error(f"   ❌ LibreTranslate failed: {e}")
            raise


class GoogleTranslateService(TranslationService):
    """Google Translate API service"""

    def __init__(self, api_key: str, logger: Logger):
        super().__init__(logger)
        self.api_key = api_key
        self.endpoint = "https://translation.googleapis.com/language/translate/v2"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Google Translate API"""
        self.logger.debug(f"   🌍 Translating with Google Translate")

        params = {
            "key": self.api_key,
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text",
        }

        try:
            response = requests.post(self.endpoint, params=params, timeout=60)
            response.raise_for_status()

            result = response.json()
            translated = result["data"]["translations"][0]["translatedText"]
            self.logger.debug(f"   ✓ Translation received")
            return translated

        except Exception as e:
            self.logger.error(f"   ❌ Google Translate failed: {e}")
            raise


class SRTTranslator:
    """Main SRT translation orchestrator"""

    def __init__(self, verbose: bool = False):
        self.logger = Logger(verbose)
        self.work_dir = Path.cwd()
        self.source_entries: List[SRTEntry] = []
        self.temp_source_file: Optional[str] = None
        self.temp_target_file: Optional[str] = None

    def load_srt(self, file_path: str) -> Tuple[List[Dict], str]:
        """Load and parse SRT file"""
        self.logger.clear()
        self.logger.info("=" * 60)
        self.logger.info("🚀 Starting SRT Translation Process")
        self.logger.info("=" * 60)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.source_entries = SRTParser.parse(content, self.logger)

            # Save temporary source file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.temp_source_file = self.work_dir / f"temp_source_{timestamp}.srt"
            SRTParser.write(
                self.source_entries, str(self.temp_source_file), self.logger
            )

            # Convert to dict for UI
            entries_dict = [entry.to_dict() for entry in self.source_entries]

            return entries_dict, self.logger.get_logs()

        except Exception as e:
            self.logger.error(f"❌ Failed to load SRT file: {e}")
            return [], self.logger.get_logs()

    def create_translation_service(
        self,
        service_type: str,
        ai_host: str = "",
        ai_key: str = "",
        ai_model: str = "",
        libre_host: str = "",
        libre_key: str = "",
        google_key: str = "",
    ) -> Optional[TranslationService]:
        """Create appropriate translation service"""

        if service_type == "OpenAI-Compatible API":
            if not all([ai_host, ai_key, ai_model]):
                self.logger.error("❌ AI service requires host, API key, and model")
                return None
            return OpenAITranslationService(ai_host, ai_key, ai_model, self.logger)

        elif service_type == "LibreTranslate":
            if not libre_host:
                self.logger.error("❌ LibreTranslate requires host URL")
                return None
            return LibreTranslateService(libre_host, libre_key, self.logger)

        elif service_type == "Google Translate":
            if not google_key:
                self.logger.error("❌ Google Translate requires API key")
                return None
            return GoogleTranslateService(google_key, self.logger)

        return None

    def translate_entries(
        self,
        service: TranslationService,
        source_lang: str,
        target_lang: str,
        translate_all: bool = True,
        entry_index: Optional[int] = None,
    ) -> Tuple[List[Dict], str]:
        """Translate subtitle entries"""

        self.logger.info("=" * 60)
        self.logger.info("🔄 Starting Translation")
        self.logger.info("=" * 60)
        self.logger.info(f"📝 Source Language: {source_lang}")
        self.logger.info(f"📝 Target Language: {target_lang}")

        try:
            if translate_all:
                total = len(self.source_entries)
                self.logger.info(f"📊 Translating {total} entries...")

                for idx, entry in enumerate(self.source_entries, 1):
                    self.logger.info(f"[{idx}/{total}] 🔄 Entry #{entry.sequence}")
                    self.logger.debug(f"   Source: {entry.text[:50]}...")

                    translated = service.translate(entry.text, source_lang, target_lang)
                    entry.translated_text = translated

                    self.logger.debug(f"   Target: {translated[:50]}...")
                    self.logger.info(f"[{idx}/{total}] ✅ Completed")

                self.logger.info("=" * 60)
                self.logger.info("🎉 Translation Complete!")
                self.logger.info("=" * 60)

            else:
                if entry_index is None or entry_index >= len(self.source_entries):
                    self.logger.error("❌ Invalid entry index")
                    return [], self.logger.get_logs()

                entry = self.source_entries[entry_index]
                self.logger.info(f"🔄 Translating entry #{entry.sequence}")

                translated = service.translate(entry.text, source_lang, target_lang)
                entry.translated_text = translated

                self.logger.info(f"✅ Entry #{entry.sequence} translated")

            # Save temporary target file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.temp_target_file = self.work_dir / f"temp_target_{timestamp}.srt"
            SRTParser.write(
                self.source_entries,
                str(self.temp_target_file),
                self.logger,
                use_translated=True,
            )

            entries_dict = [entry.to_dict() for entry in self.source_entries]
            return entries_dict, self.logger.get_logs()

        except Exception as e:
            self.logger.error(f"❌ Translation failed: {e}")
            return [], self.logger.get_logs()

    def save_final_translation(self, filename: str = None) -> str:
        """Save final translated SRT file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"translated_{timestamp}.srt"

        output_path = self.work_dir / filename

        self.logger.info("=" * 60)
        SRTParser.write(
            self.source_entries, str(output_path), self.logger, use_translated=True
        )
        self.logger.info("=" * 60)

        return str(output_path)


def create_gradio_interface():
    """Create the Gradio UI"""

    translator = SRTTranslator()

    # Language options (common languages)
    languages = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "ru",
        "ja",
        "ko",
        "zh",
        "ar",
        "hi",
        "nl",
        "pl",
        "tr",
        "sv",
        "da",
        "no",
        "fi",
        "cs",
    ]

    with gr.Blocks(title="SRT Translator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 SRT File Translator")
        gr.Markdown("Translate subtitle files using AI or translation services")

        with gr.Row():
            # Left Column - Service Configuration
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Translation Service")

                service_type = gr.Dropdown(
                    choices=[
                        "OpenAI-Compatible API",
                        "LibreTranslate",
                        "Google Translate",
                    ],
                    label="Service Type",
                    value="OpenAI-Compatible API",
                )

                # AI Service Config
                with gr.Group(visible=True) as ai_config:
                    ai_host = gr.Textbox(
                        label="API Host", placeholder="http://localhost:11434"
                    )
                    ai_key = gr.Textbox(
                        label="API Key", placeholder="sk-...", type="password"
                    )
                    ai_model = gr.Textbox(label="Model", placeholder="gpt-4")

                # LibreTranslate Config
                with gr.Group(visible=False) as libre_config:
                    libre_host = gr.Textbox(
                        label="LibreTranslate Host",
                        placeholder="https://libretranslate.com",
                    )
                    libre_key = gr.Textbox(label="API Key (optional)", type="password")

                # Google Translate Config
                with gr.Group(visible=False) as google_config:
                    google_key = gr.Textbox(label="Google API Key", type="password")

                verbose_logging = gr.Checkbox(
                    label="Enable Verbose Logging", value=False
                )

            # Middle Column - Source
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Source File")

                source_lang = gr.Dropdown(
                    choices=languages, label="Source Language", value="en"
                )

                file_input = gr.File(
                    label="Upload SRT File", file_types=[".srt"], type="filepath"
                )

                load_btn = gr.Button("Load File", variant="primary")

                source_editor = gr.Dataframe(
                    headers=["Seq", "Timecode", "Text"],
                    label="Source Subtitles (Editable)",
                    interactive=True,
                    wrap=True,
                )

            # Right Column - Target
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Translation")

                target_lang = gr.Dropdown(
                    choices=languages, label="Target Language", value="es"
                )

                translate_all_btn = gr.Button(
                    "🌍 Translate All", variant="primary", size="lg"
                )

                target_editor = gr.Dataframe(
                    headers=["Seq", "Timecode", "Translated Text"],
                    label="Translated Subtitles (Editable)",
                    interactive=True,
                    wrap=True,
                )

                download_btn = gr.Button("⬇️ Download Translation", variant="secondary")
                output_file = gr.File(label="Download")

        # Bottom Row - Logs
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Process Logs")
                log_output = gr.Textbox(
                    label="Logs", lines=15, max_lines=20, interactive=False
                )

        # Event Handlers
        def toggle_service_config(service):
            return (
                gr.update(visible=service == "OpenAI-Compatible API"),
                gr.update(visible=service == "LibreTranslate"),
                gr.update(visible=service == "Google Translate"),
            )

        service_type.change(
            toggle_service_config,
            inputs=[service_type],
            outputs=[ai_config, libre_config, google_config],
        )

        def load_file(file_path, verbose):
            if not file_path:
                return None, None, "⚠️ Please select a file"

            translator.logger.verbose = verbose
            entries, logs = translator.load_srt(file_path)

            # Format for display
            source_data = [[e["sequence"], e["timecode"], e["text"]] for e in entries]

            return source_data, None, logs

        load_btn.click(
            load_file,
            inputs=[file_input, verbose_logging],
            outputs=[source_editor, target_editor, log_output],
        )

        def translate_all(
            service,
            ai_h,
            ai_k,
            ai_m,
            libre_h,
            libre_k,
            google_k,
            src_lang,
            tgt_lang,
            verbose,
        ):
            translator.logger.verbose = verbose

            service_obj = translator.create_translation_service(
                service, ai_h, ai_k, ai_m, libre_h, libre_k, google_k
            )

            if not service_obj:
                return None, translator.logger.get_logs()

            entries, logs = translator.translate_entries(
                service_obj, src_lang, tgt_lang, translate_all=True
            )

            # Format for display
            target_data = [
                [e["sequence"], e["timecode"], e["translated_text"]] for e in entries
            ]

            return target_data, logs

        translate_all_btn.click(
            translate_all,
            inputs=[
                service_type,
                ai_host,
                ai_key,
                ai_model,
                libre_host,
                libre_key,
                google_key,
                source_lang,
                target_lang,
                verbose_logging,
            ],
            outputs=[target_editor, log_output],
        )

        def save_translation():
            output_path = translator.save_final_translation()
            return output_path, translator.logger.get_logs()

        download_btn.click(save_translation, outputs=[output_file, log_output])

    return app


if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
