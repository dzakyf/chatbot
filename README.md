# spaCy Chatbot 

A Python + spaCy chatbot backend. For learn purpose only.

## Project Structure

```
chatbot/
├── core/
│   ├── __init__.py
│   ├── nlu.py          # Natural Language Understanding (intent + entity)
│   ├── dialogue.py     # Dialogue Manager (conversation state)
│   └── response.py     # Response Generator
├── data/
│   └── intents.json    # Intent patterns and responses
├── tests/
│   └── test_nlu.py
├── main.py             # CLI entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
python main.py
```

## Test

```bash
python -m pytest tests/
```

## How it works

```
User Input
    
1. NLU (nlu.py)
  - Tokenize with spaCy
  - Match intent via vector similarity
  - Extract named entities (NER)
    
2. DialogueManager (dialogue.py)
  - Update context / state
  - Coordinate NLU + Responder
    
3. ResponseGenerator (response.py)
  - Pick response for matched intent
    
4. Bot Response
```
