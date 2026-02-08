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
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import requests


class Logger:
    """Pretty logger for translation process with streaming support"""

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

    def translate(
        self, text: str, source_lang: str, target_lang: str, max_retries: int = 5
    ) -> str:
        """Translate text from source to target language with retry logic"""
        raise NotImplementedError


class OpenAITranslationService(TranslationService):
    """OpenAI-compatible API translation service"""

    def __init__(self, host: str, api_key: str, model: str, logger: Logger):
        super().__init__(logger)
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.model = model

    def translate(
        self, text: str, source_lang: str, target_lang: str, max_retries: int = 5
    ) -> str:
        """Translate using OpenAI-compatible API with retry logic"""
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

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.host}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()

                result = response.json()
                translated = result["choices"][0]["message"]["content"].strip()
                self.logger.debug(f"   ✓ Translation received")
                return translated

            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"   ⚠️  Timeout on attempt {attempt + 1}/{max_retries}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise

            except Exception as e:
                self.logger.warning(
                    f"   ⚠️  Error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(
                        f"   ❌ AI translation failed after {max_retries} attempts"
                    )
                    raise


class LibreTranslateService(TranslationService):
    """LibreTranslate API service"""

    def __init__(self, host: str, api_key: str, logger: Logger):
        super().__init__(logger)
        self.host = host.rstrip("/")
        self.api_key = api_key

    def translate(
        self, text: str, source_lang: str, target_lang: str, max_retries: int = 5
    ) -> str:
        """Translate using LibreTranslate with retry logic"""
        self.logger.debug(f"   🌐 Translating with LibreTranslate")

        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text",
        }

        if self.api_key:
            payload["api_key"] = self.api_key

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.host}/translate", json=payload, timeout=120
                )
                response.raise_for_status()

                result = response.json()
                translated = result["translatedText"]
                self.logger.debug(f"   ✓ Translation received")
                return translated

            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"   ⚠️  Timeout on attempt {attempt + 1}/{max_retries}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise

            except Exception as e:
                self.logger.warning(
                    f"   ⚠️  Error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(
                        f"   ❌ LibreTranslate failed after {max_retries} attempts"
                    )
                    raise


class GoogleTranslateService(TranslationService):
    """Google Translate API service"""

    def __init__(self, api_key: str, logger: Logger):
        super().__init__(logger)
        self.api_key = api_key
        self.endpoint = "https://translation.googleapis.com/language/translate/v2"

    def translate(
        self, text: str, source_lang: str, target_lang: str, max_retries: int = 5
    ) -> str:
        """Translate using Google Translate API with retry logic"""
        self.logger.debug(f"   🌍 Translating with Google Translate")

        params = {
            "key": self.api_key,
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text",
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.endpoint, params=params, timeout=120)
                response.raise_for_status()

                result = response.json()
                translated = result["data"]["translations"][0]["translatedText"]
                self.logger.debug(f"   ✓ Translation received")
                return translated

            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"   ⚠️  Timeout on attempt {attempt + 1}/{max_retries}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise

            except Exception as e:
                self.logger.warning(
                    f"   ⚠️  Error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(
                        f"   ❌ Google Translate failed after {max_retries} attempts"
                    )
                    raise


class SRTTranslator:
    """Main SRT translation orchestrator"""

    def __init__(self, verbose: bool = False):
        self.logger = Logger(verbose)
        self.work_dir = Path.cwd()
        self.source_entries: List[SRTEntry] = []
        self.temp_source_file: Optional[str] = None
        self.temp_target_file: Optional[str] = None
        self.stop_event = Event()
        self.current_index = 0

    def load_srt(self, file_path: str) -> Tuple[pd.DataFrame, str]:
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

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    [e.sequence, e.timecode, e.text, e.translated_text]
                    for e in self.source_entries
                ],
                columns=["Seq", "Timecode", "Text", "Translated"],
            )

            return df, self.logger.get_logs()

        except Exception as e:
            self.logger.error(f"❌ Failed to load SRT file: {e}")
            return pd.DataFrame(
                columns=["Seq", "Timecode", "Text", "Translated"]
            ), self.logger.get_logs()

    def sync_from_dataframe(self, dataframe_data):
        """Sync entries from dataframe - called automatically on any table change"""
        # Handle both DataFrame and list inputs
        if isinstance(dataframe_data, pd.DataFrame):
            if dataframe_data.empty:
                return
            data = dataframe_data.values.tolist()
        else:
            data = dataframe_data

        if not data:
            return

        # Rebuild entries with automatic renumbering
        new_entries = []
        for idx, row in enumerate(data, 1):
            if len(row) >= 3:
                # row format: [sequence, timecode, text, translated_text]
                timecode = str(row[1]) if row[1] else "00:00:00,000 --> 00:00:00,000"
                text = str(row[2]) if row[2] else ""
                entry = SRTEntry(idx, timecode, text)
                if len(row) >= 4 and row[3]:
                    entry.translated_text = str(row[3])
                new_entries.append(entry)

        self.source_entries = new_entries
        self.logger.info(f"✅ Synced {len(new_entries)} entries from table")

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

    def stop_translation(self):
        """Stop the translation process"""
        self.stop_event.set()
        self.logger.warning("⏹️  Translation stopped by user")
        app.get

    def translate_single_entry(
        self,
        service: TranslationService,
        entry_index: int,
        source_lang: str,
        target_lang: str,
    ) -> Tuple[str, str]:
        """Translate a single entry"""
        try:
            if entry_index >= len(self.source_entries):
                self.logger.error(f"❌ Invalid entry index: {entry_index}")
                return "", self.logger.get_logs()

            entry = self.source_entries[entry_index - 1]
            self.logger.info(f"🔄 Translating entry #{entry.sequence}")
            self.logger.debug(f"   Source: {entry.text[:50]}...")

            translated = service.translate(entry.text, source_lang, target_lang)
            entry.translated_text = translated

            self.logger.debug(f"   Target: {translated[:50]}...")
            self.logger.info(f"✅ Entry #{entry.sequence} translated")

            return translated, self.logger.get_logs()

        except Exception as e:
            self.logger.error(f"❌ Translation failed: {e}")
            return "", self.logger.get_logs()

    def translate_all_entries(
        self,
        service: TranslationService,
        source_lang: str,
        target_lang: str,
        start_index: int = 0,
        progress=gr.Progress(),
    ) -> Tuple[pd.DataFrame, str, int]:
        """Translate all subtitle entries with pause/stop support and real-time updates"""

        self.logger.info("=" * 60)
        self.logger.info("🔄 Starting Translation")
        self.logger.info("=" * 60)
        self.logger.info(f"📝 Source Language: {source_lang}")
        self.logger.info(f"📝 Target Language: {target_lang}")

        self.stop_event.clear()
        total = len(self.source_entries)

        try:
            for idx in range(start_index, total):
                # Check if stopped
                if self.stop_event.is_set():
                    self.logger.warning("⏹️  Translation stopped")
                    self.current_index = idx
                    break

                entry = self.source_entries[idx]

                # Update progress
                progress((idx + 1) / total, desc=f"Translating entry {idx + 1}/{total}")

                self.logger.info(f"[{idx + 1}/{total}] 🔄 Entry #{entry.sequence}")
                self.logger.debug(f"   Source: {entry.text[:50]}...")

                try:
                    translated = service.translate(entry.text, source_lang, target_lang)
                    entry.translated_text = translated

                    self.logger.debug(f"   Target: {translated[:50]}...")
                    self.logger.info(f"[{idx + 1}/{total}] ✅ Completed")
                    # self.current_index = idx + 1

                    # Yield intermediate results for streaming
                    # yield (
                    #     self._get_dataframe(True),
                    #     self.logger.get_logs(),
                    #     self.current_index,
                    # )

                except Exception as e:
                    self.logger.error(f"[{idx + 1}/{total}] ❌ Failed: {e}")
                    self.logger.warning(
                        f"⚠️  Stopping at entry {idx + 1}. Already translated: {idx} entries"
                    )
                    self.current_index = idx
                    break

            if self.current_index >= total:
                self.logger.info("=" * 60)
                self.logger.info("🎉 Translation Complete!")
                self.logger.info("=" * 60)

            # Save temporary target file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.temp_target_file = self.work_dir / f"temp_target_{timestamp}.srt"
            SRTParser.write(
                self.source_entries,
                str(self.temp_target_file),
                self.logger,
                use_translated=True,
            )

            print("yielding final results")
            yield self._get_dataframe(), self.logger.get_logs(), self.current_index

        except Exception as e:
            self.logger.error(f"❌ Translation failed: {e}")
            yield self._get_dataframe(), self.logger.get_logs(), self.current_index

    def _get_dataframe(self, isFromLoop: bool = False) -> pd.DataFrame:
        """Get current entries as DataFrame"""
        if isFromLoop:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                [e.sequence, e.timecode, e.text, e.translated_text]
                for e in self.source_entries
            ],
            columns=["Seq", "Timecode", "Text", "Translated"],
        )

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

    log_output: gr.Textbox

    with gr.Blocks(title="SRT Translator") as app:
        gr.Markdown("# 🎬 SRT File Translator")
        gr.Markdown("Translate subtitle files using AI or translation services")

        # State to track if translation is running
        translation_state = gr.State({"running": False, "start_index": 0})

        with gr.Row():
            # Left Column - Service Configuration (25%)
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
                        label="API Host", placeholder="http://localhost:11434/v1"
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

            # Middle Column - Source (50%)
            with gr.Column(scale=3):
                gr.Markdown("### 📥 Source & Translation")

                with gr.Row():
                    source_lang = gr.Dropdown(
                        choices=languages, label="Source Language", value="en", scale=1
                    )
                    target_lang = gr.Dropdown(
                        choices=languages, label="Target Language", value="es", scale=1
                    )

                file_input = gr.File(
                    label="Upload SRT File", file_types=[".srt"], type="filepath"
                )

                load_btn = gr.Button("Load File", variant="primary")

                source_editor = gr.Dataframe(
                    headers=["Seq", "Timecode", "Text", "Translated"],
                    label="Subtitles (Editable - Sequences auto-update)",
                    interactive=True,
                    col_count=(4, "fixed"),
                    row_count=(10, "dynamic"),
                    buttons=["fullscreen", "copy"],
                )

                gr.Markdown(
                    "ℹ️ *Add/remove rows directly. Sequences renumber automatically on any change.*"
                )

            # Right Column - Controls (25%)
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Translation Controls")

                with gr.Row():
                    translate_all_btn = gr.Button("🌍 Translate All", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")

                continue_btn = gr.Button("🔄 Continue from Stop", variant="secondary")

                gr.Markdown("#### Individual Translation")
                entry_index = gr.Number(label="Sequence #", value=1, precision=0)
                translate_single_btn = gr.Button("🔄 Translate", variant="secondary")

                gr.Markdown("---")
                download_btn = gr.Button(
                    "⬇️ Download Translation", variant="primary", size="lg"
                )
                output_file = gr.File(label="Download")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Process Logs")
                log_output = gr.Textbox(
                    label="Logs",
                    lines=12,
                    max_lines=15,
                    interactive=False,
                    autoscroll=True,
                    show_label=False,
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
                return (
                    pd.DataFrame(columns=["Seq", "Timecode", "Text", "Translated"]),
                    "⚠️ Please select a file",
                    {"running": False, "start_index": 0},
                )

            translator.logger.verbose = verbose
            df, logs = translator.load_srt(file_path)

            return df, logs, {"running": False, "start_index": 0}

        load_btn.click(
            load_file,
            inputs=[file_input, verbose_logging],
            outputs=[source_editor, log_output, translation_state],
        )

        def on_dataframe_change(dataframe_data):
            """Auto-sync when user edits the table"""
            translator.sync_from_dataframe(dataframe_data)
            # Return updated dataframe with renumbered sequences
            return translator._get_dataframe()

        source_editor.change(
            on_dataframe_change, inputs=[source_editor], outputs=[source_editor]
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
            state,
            progress=gr.Progress(),
        ):
            translator.logger.verbose = verbose

            service_obj = translator.create_translation_service(
                service, ai_h, ai_k, ai_m, libre_h, libre_k, google_k
            )

            if not service_obj:
                return (
                    translator._get_dataframe(),
                    "",
                    translator.logger.get_logs(),
                    {"running": False, "start_index": 0},
                )

            start_idx = state.get("start_index", 0)

            # Use generator for streaming updates
            for df, logs, current_idx in translator.translate_all_entries(
                service_obj,
                src_lang,
                tgt_lang,
                start_index=start_idx,
                progress=progress,
            ):
                yield (
                    df,
                    logs,
                    {"running": False, "start_index": current_idx},
                )

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
                translation_state,
            ],
            outputs=[source_editor, log_output, translation_state],
            show_progress="full",
            show_progress_on=[source_editor],
        )

        def stop_translation():
            translator.stop_translation()
            return translator.logger.get_logs()

        stop_btn.click(stop_translation, outputs=[log_output])

        def continue_translation(
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
            state,
            progress=gr.Progress(),
        ):
            translator.logger.verbose = verbose

            service_obj = translator.create_translation_service(
                service, ai_h, ai_k, ai_m, libre_h, libre_k, google_k
            )

            if not service_obj:
                return (
                    translator._get_dataframe(),
                    "",
                    translator.logger.get_logs(),
                    state,
                )

            # Continue from where we stopped with streaming
            for df, logs, current_idx in translator.translate_all_entries(
                service_obj,
                src_lang,
                tgt_lang,
                start_index=translator.current_index,
                progress=progress,
            ):
                yield (
                    df,
                    logs,
                    {"running": False, "start_index": current_idx},
                )

        continue_btn.click(
            continue_translation,
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
                translation_state,
            ],
            outputs=[source_editor, log_output, translation_state],
        )

        def translate_single(
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
            idx,
        ):
            translator.logger.verbose = verbose

            service_obj = translator.create_translation_service(
                service, ai_h, ai_k, ai_m, libre_h, libre_k, google_k
            )

            if not service_obj:
                return translator._get_dataframe(), translator.logger.get_logs()

            translated, logs = translator.translate_single_entry(
                service_obj, int(idx), src_lang, tgt_lang
            )

            return translator._get_dataframe(), logs

        translate_single_btn.click(
            translate_single,
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
                entry_index,
            ],
            outputs=[source_editor, log_output],
        )

        def save_translation():
            output_path = translator.save_final_translation()
            return output_path, translator.logger.get_logs()

        download_btn.click(save_translation, outputs=[output_file, log_output])

    return app


if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(
        theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860, share=False
    )
