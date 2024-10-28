from llama_cpp import Llama

SYSTEM_INSTRUCTIONS = 'Your name is Zora, you are a helpful assistant, reply with BRIEF responses (max 25 words).'
MAX_HISTORY_LENGHT = 6

class Response_generator:

    def __init__(self):
        model_path = 'models/llama-2-7b-chat.Q2_K.gguf'
        self.llm = Llama(model_path=model_path, chat_format='llama-2')
        self.history = []
    
    def generate_response(self, prompt):

        self.history.append(('user', prompt))

        messages = [{"role": 'system', "content": SYSTEM_INSTRUCTIONS}]
        for role, content in self.history:
            messages.append({"role": role, "content": content})

        response = self.llm.create_chat_completion(
            messages = messages
        )
        reply = response['choices'][0]['message']['content']
        reply = reply.lstrip()

        self.history.append(('assistant', reply))
        if len(self.history) > MAX_HISTORY_LENGHT:
            self.history.pop(0)
            self.history.pop(0)

        return reply