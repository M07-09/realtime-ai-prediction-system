"""
The Transformer chatbot - the part that SPEAKS.

The numbers come from app/chat_facts.py; this module only turns them into
fluent English with a small decoder-only Transformer (Qwen2.5-0.5B-Instruct),
then refuses to publish anything that contradicts the data.

    question
       |
    detect_intent()      ->  which facts matter
    build_data_block()   ->  the verified numbers
    deterministic_answer() -> the exact answer (also the fallback)
       |
    Transformer rewrites it conversationally (greedy decoding)
       |
    verify_answer()      ->  accepted, or replaced by the exact answer
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.chat_facts import (
    STRICT_INTENTS,
    SUGGESTED_QUESTIONS,
    build_data_block,
    detect_intent,
    deterministic_answer,
    verify_answer,
)
from app.config import (
    CHATBOT_DEVICE,
    CHATBOT_HISTORY_TURNS,
    CHATBOT_MAX_NEW_TOKENS,
    CHATBOT_MODEL,
    CHATBOT_TEMPERATURE,
    CHATBOT_TOP_P,
)
from app.utils import get_logger

log = get_logger("chatbot", "chatbot.log")

SYSTEM_PROMPT = (
    "You are the assistant of a real-time Bitcoin monitoring dashboard. "
    "You will be given a DATA block containing verified live numbers and an LSTM "
    "forecast. Answer the user's question in 1-3 short sentences using ONLY the "
    "numbers in the DATA block. Always include the relevant figures, and when the "
    "question is about direction mention both the recent trend and the forecast. "
    "Never invent a price, never give financial advice, and if the DATA block does "
    "not contain the answer, say so plainly."
)

__all__ = ["TransformerChatbot", "get_chatbot", "SUGGESTED_QUESTIONS"]


# --------------------------------------------------------------------------
# Stage 3 - the Transformer
# --------------------------------------------------------------------------
class TransformerChatbot:
    """Lazily-loaded causal Transformer that phrases the grounded answer."""

    def __init__(self, model_name: str = CHATBOT_MODEL) -> None:
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.loaded = False
        self.load_error: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- loading
    def load(self) -> bool:
        """Load the weights once. Returns True on success, never raises."""
        if self.loaded:
            return True
        with self._lock:
            if self.loaded:
                return True
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                if CHATBOT_DEVICE == "cpu":
                    device = "cpu"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

                log.info("Loading Transformer %s on %s ...", self.model_name, device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                dtype = torch.float16 if device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, dtype=dtype
                )
                self.model.to(device)
                self.model.eval()
                self.device = device
                self.loaded = True
                self.load_error = None
                log.info("Transformer ready on %s", device)
                return True
            except Exception as exc:                  # noqa: BLE001
                self.load_error = f"{type(exc).__name__}: {exc}"
                log.error("Could not load the Transformer: %s", self.load_error)
                return False

    # ---------------------------------------------------------- generation
    def _generate(self, messages: List[Dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Greedy decoding by default: reporting live data must be reproducible.
        # Sampling flags are only passed when sampling is actually enabled,
        # otherwise transformers warns that they are ignored.
        options: Dict[str, Any] = {
            "max_new_tokens": CHATBOT_MAX_NEW_TOKENS,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if CHATBOT_TEMPERATURE > 0:
            options.update(do_sample=True, temperature=CHATBOT_TEMPERATURE, top_p=CHATBOT_TOP_P)
        else:
            options["do_sample"] = False

        with torch.no_grad():
            output = self.model.generate(**inputs, **options)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ---------------------------------------------------------------- API
    def answer(
        self,
        question: str,
        context: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Return a grounded answer plus the metadata the dashboard displays."""
        question = (question or "").strip()
        if not question:
            return {
                "answer": "Please type a question about the live data or the forecast.",
                "intent": "empty",
                "engine": "rules",
                "grounded_facts": "",
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

        intent = detect_intent(question)
        data_block = build_data_block(context, intent)
        fallback = deterministic_answer(context, intent)

        engine = "rules (Transformer unavailable)"
        answer = fallback

        if self.load():
            try:
                messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
                for turn in (history or [])[-CHATBOT_HISTORY_TURNS * 2:]:
                    role = turn.get("role")
                    content = (turn.get("content") or "").strip()
                    if role in {"user", "assistant"} and content:
                        messages.append({"role": role, "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"DATA (verified live numbers):\n{data_block}\n\n"
                        f"VERIFIED ANSWER: {fallback}\n\n"
                        f"QUESTION: {question}\n\n"
                        "Rewrite the VERIFIED ANSWER as a natural, conversational reply to the "
                        "QUESTION. Keep every number and every fact it contains, change no "
                        "value, and add nothing that is not in the DATA."
                    ),
                })
                generated = self._generate(messages)
                if generated and len(generated) > 3:
                    ok, reason = verify_answer(
                        generated, data_block, context,
                        required_from=fallback if intent in STRICT_INTENTS else None,
                    )
                    if ok:
                        answer = generated
                        engine = f"{self.model_name} on {self.device}"
                    else:
                        # The Transformer drifted from the data: keep the exact
                        # answer instead, and say why.
                        log.warning("Generated answer rejected (%s): %s", reason, generated[:120])
                        engine = f"rules (generated answer rejected: {reason})"
            except Exception as exc:                  # noqa: BLE001
                log.warning("Generation failed, falling back to the rule answer: %s", exc)
                engine = f"rules (generation error: {type(exc).__name__})"

        return {
            "answer": answer,
            "deterministic_answer": fallback,
            "intent": intent,
            "engine": engine,
            "grounded_facts": data_block,
            "model_loaded": self.loaded,
            "model_error": self.load_error,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


_chatbot: Optional[TransformerChatbot] = None


def get_chatbot() -> TransformerChatbot:
    global _chatbot
    if _chatbot is None:
        _chatbot = TransformerChatbot()
    return _chatbot
