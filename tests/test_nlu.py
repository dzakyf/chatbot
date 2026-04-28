"""
tests/test_nlu.py - Basic tests for the NLU layer.
Run with: python -m pytest tests/
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---- NLU Tests ----

def test_nlu_loads():
    from core.nlu import NLU
    nlu = NLU(intents_path="data/intents.json")
    assert nlu is not None
    assert len(nlu.intents) > 0


def test_parse_returns_expected_keys():
    from core.nlu import NLU
    nlu = NLU(intents_path="data/intents.json")
    result = nlu.parse("hello there")
    assert "intent" in result
    assert "confidence" in result
    assert "entities" in result
    assert "tokens" in result


def test_unknown_intent_for_gibberish():
    from core.nlu import NLU
    nlu = NLU(intents_path="data/intents.json")
    result = nlu.parse("asdfghjkl qwerty")
    # Should not crash, may return unknown
    assert isinstance(result["intent"], str)


# ---- Dialogue Tests ----

def test_dialogue_returns_string():
    from core.dialogue import DialogueManager
    bot = DialogueManager(intents_path="data/intents.json")
    response = bot.process("hello")
    assert isinstance(response, str)
    assert len(response) > 0


def test_dialogue_logs_history():
    from core.dialogue import DialogueManager
    bot = DialogueManager(intents_path="data/intents.json")
    bot.process("hi")
    bot.process("bye")
    history = bot.get_history()
    assert len(history) == 2


def test_dialogue_reset():
    from core.dialogue import DialogueManager
    bot = DialogueManager(intents_path="data/intents.json")
    bot.process("hello")
    bot.reset()
    assert len(bot.history) == 0
    assert bot.context == {}


def test_empty_input():
    from core.dialogue import DialogueManager
    bot = DialogueManager(intents_path="data/intents.json")
    response = bot.process("")
    assert isinstance(response, str)
