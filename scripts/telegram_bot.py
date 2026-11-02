"""
Production Telegram Bot Integration for Agentic RAG Microservice.

This lightweight client uses httpx/requests to interface with the arXiv RAG API,
supporting long-polling without requiring heavy external bot frameworks.
"""

import argparse
import html
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("telegram_rag_bot")

TELEGRAM_API_BASE = "https://api.telegram.org/bot"


class TelegramRAGBot:
    """Telegram Bot client interacting with the Agentic RAG Microservice."""

    def __init__(
        self,
        bot_token: str,
        api_base_url: str = "http://localhost:8000/api/v1",
        default_model: str = "llama3.2:1b",
    ):
        self.bot_token = bot_token
        self.api_base_url = api_base_url.rstrip("/")
        self.default_model = default_model
        self.telegram_url = f"{TELEGRAM_API_BASE}{self.bot_token}"
        self.offset = 0

    def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
        """Send message back to Telegram user."""
        url = f"{self.telegram_url}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if not resp.ok:
                # Fallback to plain text if HTML parsing failed
                payload["parse_mode"] = ""
                requests.post(url, json=payload, timeout=15)
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Send typing indicator."""
        url = f"{self.telegram_url}/sendChatAction"
        try:
            requests.post(url, json={"chat_id": chat_id, "action": action}, timeout=5)
        except Exception:
            pass

    def check_api_health(self) -> Dict[str, Any]:
        """Query /api/v1/health endpoint."""
        url = f"{self.api_base_url}/health"
        try:
            resp = requests.get(url, timeout=10)
            return resp.json() if resp.ok else {"status": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def query_agentic_rag(self, query: str, top_k: int = 3, use_hybrid: bool = True) -> Dict[str, Any]:
        """Query /api/v1/agentic_ask endpoint."""
        url = f"{self.api_base_url}/agentic_ask"
        payload = {
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "model": self.default_model,
        }
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()

    def query_standard_rag(self, query: str, top_k: int = 3, use_hybrid: bool = True) -> Dict[str, Any]:
        """Query /api/v1/ask endpoint."""
        url = f"{self.api_base_url}/ask"
        payload = {
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "model": self.default_model,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def format_agentic_response(self, data: Dict[str, Any]) -> str:
        """Format agentic response with HTML styling for Telegram."""
        answer = data.get("answer", "No answer generated.")
        steps = data.get("reasoning_steps", [])
        sources = data.get("sources", [])
        rewritten = data.get("rewritten_query")
        attempts = data.get("retrieval_attempts", 0)

        lines = [f"<b>Research Answer</b>\n", html.escape(answer)]

        if rewritten:
            lines.append(f"\n<i>Expanded Query:</i> <code>{html.escape(rewritten)}</code>")

        if steps:
            lines.append("\n<b>Reasoning Steps:</b>")
            for step in steps:
                lines.append(f"• <i>{html.escape(step)}</i>")

        if sources:
            lines.append(f"\n<b>Sources ({len(sources)} papers):</b>")
            for i, src in enumerate(sources[:5], 1):
                if isinstance(src, str):
                    paper_id = src.split("/")[-1].replace(".pdf", "")
                    lines.append(f"{i}. <a href=\"{html.escape(src)}\">arXiv:{html.escape(paper_id)}</a>")
                elif isinstance(src, dict):
                    title = src.get("title", "Paper")
                    url = src.get("url", "#")
                    lines.append(f"{i}. <a href=\"{html.escape(url)}\">{html.escape(title)}</a>")

        return "\n".join(lines)

    def handle_message(self, message: Dict[str, Any]) -> None:
        """Process incoming user message."""
        chat_id = message["chat"]["id"]
        text = message.get("text", "").strip()

        if not text:
            return

        logger.info(f"Received message from chat_id={chat_id}: {text}")

        # Command handling
        if text.startswith("/start"):
            welcome = (
                "👋 <b>Welcome to arXiv AI Research Assistant!</b>\n\n"
                "I can search thousands of arXiv AI & ML papers and answer complex technical questions "
                "with citations and agentic reasoning.\n\n"
                "<b>Commands:</b>\n"
                "• <code>/ask &lt;query&gt;</code> - Fast RAG retrieval & answer\n"
                "• <code>/agentic &lt;query&gt;</code> - Agentic RAG with guardrails & query expansion\n"
                "• <code>/status</code> - Check backend microservice health\n"
                "• <code>/help</code> - Show instructions\n\n"
                "Or simply send your research question directly!"
            )
            self.send_message(chat_id, welcome)
            return

        if text.startswith("/help"):
            help_text = (
                "📖 <b>How to Use:</b>\n\n"
                "1. Send any Machine Learning / AI question (e.g. <i>'How does LoRA fine-tuning work?'</i>)\n"
                "2. The system checks domain relevance, optimizes the query, retrieves vector + BM25 chunks, "
                "and synthesizes an answer with direct arXiv PDF links.\n\n"
                "<b>Backend API:</b> " + html.escape(self.api_base_url)
            )
            self.send_message(chat_id, help_text)
            return

        if text.startswith("/status"):
            self.send_chat_action(chat_id, "typing")
            health = self.check_api_health()
            status_text = (
                f"<b>Microservice Status:</b> <code>{health.get('status', 'unknown')}</code>\n"
                f"<b>Service:</b> <code>{health.get('service_name', 'rag-api')}</code>\n"
                f"<b>Version:</b> <code>{health.get('version', '0.1.0')}</code>"
            )
            self.send_message(chat_id, status_text)
            return

        # Query handling
        use_standard = False
        query = text
        if text.startswith("/ask "):
            use_standard = True
            query = text[5:].strip()
        elif text.startswith("/agentic "):
            query = text[9:].strip()

        self.send_chat_action(chat_id, "typing")

        try:
            if use_standard:
                res = self.query_standard_rag(query)
                answer = res.get("answer", "No answer found.")
                sources = res.get("sources", [])
                out = f"<b>Answer:</b>\n{html.escape(answer)}\n\n<b>Sources:</b> {len(sources)} papers"
                self.send_message(chat_id, out)
            else:
                res = self.query_agentic_rag(query)
                out = self.format_agentic_response(res)
                self.send_message(chat_id, out)
        except requests.exceptions.HTTPError as e:
            err_msg = f"⚠️ API Error ({e.response.status_code}): {html.escape(e.response.text[:200])}"
            self.send_message(chat_id, err_msg)
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            self.send_message(chat_id, f"❌ Error contacting RAG backend: {html.escape(str(e))}")

    def run(self) -> None:
        """Run long-polling loop."""
        logger.info("Starting Telegram Bot listener...")
        while True:
            try:
                url = f"{self.telegram_url}/getUpdates"
                params = {"offset": self.offset, "timeout": 30}
                resp = requests.get(url, params=params, timeout=40)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("result", []):
                        self.offset = item["update_id"] + 1
                        if "message" in item:
                            self.handle_message(item["message"])
                else:
                    logger.warning(f"Failed getUpdates: HTTP {resp.status_code}")
                    time.sleep(2)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"Polling loop error: {e}")
                time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="arXiv RAG Telegram Bot Service")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN"), help="Telegram Bot Token")
    parser.add_argument("--api-url", default=os.getenv("RAG_API_URL", "http://localhost:8000/api/v1"), help="RAG API Base URL")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama LLM model")
    args = parser.parse_args()

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("ERROR: Telegram Bot Token not provided. Set TELEGRAM_BOT_TOKEN env or pass --token.")
        sys.exit(1)

    bot = TelegramRAGBot(bot_token=token, api_base_url=args.api_url, default_model=args.model)
    bot.run()


if __name__ == "__main__":
    main()

