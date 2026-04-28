"""
dialogue.py - Dialogue Manager
Tracks conversation history and manages turn-taking.
This is the "brain" that coordinates NLU + Response.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from core.nlu import NLU
from core.response import ResponseGenerator
from langdetect import detect, LangDetectException


@dataclass
class Turn:
    """A single exchange in the conversation."""
    user_input: str
    intent: str
    confidence: float
    entities: list[dict]
    bot_response: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

MAX_HISTORY = 20


class DialogueManager:
    def __init__(self, intents_path: str = "data/intents.json", max_history: int = MAX_HISTORY):
        self.nlu = NLU(intents_path=intents_path)
        self.responder = ResponseGenerator(intents_path=intents_path)
        self.history: deque[Turn] = deque(maxlen=max_history)
        self.context: dict = {}

    def process(self, user_input: str) -> str:
        """
        Main entry point. Takes raw user text, returns bot response.

        Args:
            user_input: raw string from user

        Returns:
            bot response string
        """
        user_input = user_input.strip()
        if not user_input:
            return "Please say something!"

        # 1. Understand
        parsed = self.nlu.parse(user_input)

        # 2. Detect language
        try:
            lang = detect(user_input)
            lang = lang if lang in ("en", "id") else "en"
        except LangDetectException:
            lang = "en"

        # 3. Update context (extend this for multi-turn flows)
        self._update_context(parsed)

        # 4. Generate response
        response = self.responder.get_response(
            intent=parsed["intent"],
            lang=lang,
            entities=parsed["entities"]
        )

        # 4. Log turn
        turn = Turn(
            user_input=user_input,
            intent=parsed["intent"],
            confidence=parsed["confidence"],
            entities=parsed["entities"],
            bot_response=response,
        )
        self.history.append(turn)

        return response

    def _update_context(self, parsed: dict):
        """
        Update conversation context based on latest parse.
        Extend this for slot-filling, multi-turn dialogs, etc.
        """
        # Example: remember the last intent
        self.context["last_intent"] = parsed["intent"]

        # Example: store any person names mentioned
        for entity in parsed.get("entities", []):
            if entity["label"] == "PERSON":
                self.context["user_name"] = entity["text"]

    def get_history(self) -> list[dict]:
        """Return conversation history as list of dicts."""
        return [
            {
                "user": t.user_input,
                "intent": t.intent,
                "confidence": t.confidence,
                "bot": t.bot_response,
                "timestamp": t.timestamp,
            }
            for t in self.history
        ]

    def reset(self):
        """Clear conversation history and context."""
        self.history = []
        self.context = {}
