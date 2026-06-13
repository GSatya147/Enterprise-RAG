class ConversationManager:
    def __init__(self):
        self.CONTEXT: list[dict] = []

    def add_context(self, message: str, role: str) -> list[dict]:
        message_dict = {"role": role, "parts": message}

        self.CONTEXT.append(message_dict)

    def get_history(self):
        return self.CONTEXT
