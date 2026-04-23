import time
from groq import Groq
from backend.app.config import settings

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
    
    def generate(self, prompt: str, system_prompt: str = None) -> dict:
        start = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=512
        )
        latency_ms = (time.time() - start) * 1000
        
        # Groq pricing for Llama 3.1 8B: $0.05 per 1M tokens
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        cost = (total_tokens / 1_000_000) * 0.05
        
        return {
            "content": response.choices[0].message.content,
            "model": self.model,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost, 6),
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        }