"""
nlu.py - Natural Language Understanding
Handles:
  - Tokenization
  - Entity extraction (NER)
  - Intent matching via spaCy similarity
"""

import json
import random
from pathlib import Path
import spacy


class NLU:
    def __init__(self, intents_path: str = "data/intents.json", model: str = "xx_ent_wiki_sm"):
        """
        Load spaCy model and intents.

        Args:
            intents_path: path to intents JSON file
            model: spaCy model name (needs to be downloaded first)
        """
        self.nlp = spacy.load(model)
        self.intents = self._load_intents(intents_path)

        # Pre-compute spaCy docs and raw patterns for matching
        self.pattern_docs = []
        self.raw_patterns: list[tuple[str, str]] = []
        for intent in self.intents:
            for pattern in intent["patterns"]:
                doc = self.nlp(pattern.lower())
                self.pattern_docs.append((doc, intent["tag"]))
                self.raw_patterns.append((pattern.lower(), intent["tag"]))

    def _load_intents(self, path: str) -> list[dict]:
        intents_file = Path(path)
        if not intents_file.exists():
            raise FileNotFoundError(f"Intents file not found: {path}")
        with open(intents_file) as f:
            data = json.load(f)
        return data["intents"]

    def parse(self, text: str) -> dict:
        """
        Parse user input and return intent + entities.

        Returns:
            {
                "text": original text,
                "intent": matched intent tag (or "unknown"),
                "confidence": similarity score 0.0 - 1.0,
                "entities": list of named entities [{text, label}]
            }
        """
        normalized = text.lower().strip()
        doc = self.nlp(normalized)

        # --- Pass 1: exact match or token overlap against raw patterns ---
        best_intent = "unknown"
        best_score = 0.0

        input_tokens = set(normalized.split())
        for pattern, tag in self.raw_patterns:
            if pattern == normalized:
                return self._build_result(text, tag, 1.0, doc)

            pattern_tokens = set(pattern.split())
            common = input_tokens & pattern_tokens
            if not common:
                continue

            # Jaccard-style: overlap / union — rewards shared words, penalises extra words
            score = len(common) / len(input_tokens | pattern_tokens)
            if score > best_score:
                best_score = score
                best_intent = tag

        if best_score >= 0.4:
            return self._build_result(text, best_intent, best_score, doc)

        # --- Pass 2: similarity / keyword overlap ---
        best_intent = "unknown"
        best_score = 0.0

        for pattern_doc, tag in self.pattern_docs:
            if doc.has_vector and pattern_doc.has_vector:
                score = doc.similarity(pattern_doc)
            else:
                score = self._keyword_overlap(doc, pattern_doc)

            if score > best_score:
                best_score = score
                best_intent = tag

        if best_score < 0.5:
            best_intent = "unknown"

        return self._build_result(text, best_intent, best_score, doc)

    def _build_result(self, text: str, intent: str, score: float, doc) -> dict:
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]
        return {
            "text": text,
            "intent": intent,
            "confidence": round(score, 3),
            "entities": entities,
            "tokens": [token.text for token in doc if not token.is_punct],
        }

    def _keyword_overlap(self, doc1, doc2) -> float:
        """Simple fallback similarity: ratio of shared lemmas."""
        lemmas1 = {token.lemma_ for token in doc1 if not token.is_stop}
        lemmas2 = {token.lemma_ for token in doc2 if not token.is_stop}
        if not lemmas1 or not lemmas2:
            return 0.0
        overlap = lemmas1 & lemmas2
        return len(overlap) / max(len(lemmas1), len(lemmas2))
