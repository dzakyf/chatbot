"""
response.py - Response Generator
Given a parsed intent, picks an appropriate response.
"""

import random
import json
from pathlib import Path


class ResponseGenerator:
    def __init__(self, intents_path: str = "data/intents.json"):
        with open(Path(intents_path)) as f:
            data = json.load(f)
        self.responses: dict[str, dict] = {
            intent["tag"]: intent["responses"]
            for intent in data["intents"]
        }
        self.fallback_responses = {
            "en": [
                "I'm not sure I understand. Could you rephrase that?",
                "Hmm, I didn't quite get that.",
                "I don't have an answer for that yet.",
            ],
            "id": [
                "Maaf, saya kurang mengerti. Bisa diulangi dengan cara lain?",
                "Hmm, saya tidak paham maksudnya.",
                "Saya belum punya jawaban untuk itu.",
            ],
        }

    def get_response(self, intent: str, lang: str = "en", entities: list[dict] = None) -> str:
        lang = lang if lang in ("en", "id") else "en"

        if intent == "unknown" or intent not in self.responses:
            return random.choice(self.fallback_responses[lang])

        pool = self.responses[intent]
        # pool is {"en": [...], "id": [...]} or a plain list (backwards compat)
        if isinstance(pool, dict):
            responses = pool.get(lang) or pool.get("en", [])
        else:
            responses = pool

        response = random.choice(responses)

        if entities:
            for entity in entities:
                if entity["label"] == "PERSON":
                    response = response.replace("{name}", entity["text"])

        return response
