"""
main.py - CLI entry point for the chatbot.
Run with: python main.py
"""

from core.dialogue import DialogueManager


def main():
    print("=" * 40)
    print("  spaCy Chatbot (type 'quit' to exit)")
    print("=" * 40)

    bot = DialogueManager(intents_path="data/intents.json")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bot: Goodbye!")
            break

        if not user_input:
            continue

        response = bot.process(user_input)
        print(f"Bot: {response}")

        # Uncomment to debug intent detection:
        # last = bot.history[-1]
        # print(f"  [intent={last.intent}, confidence={last.confidence}]")


if __name__ == "__main__":
    main()
