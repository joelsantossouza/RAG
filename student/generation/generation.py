from ..models import Chunk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME: str = "Qwen/Qwen3-0.6B"


class AnswerGenerator:
    """
    """

    def __init__(self) -> None:
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if self.device in ("cuda", "mps")
            else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

    def generate_answer(self, question: str, resource: list[Chunk]) -> str:
        context: str = "\n\n".join(chunk.data for chunk in resource)
        prompt: str = f"""
        You are a helpful assistant.

        Answer the question using ONLY the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200
        )

        generated_tokens = outputs[0][
            inputs.input_ids.shape[1]:
        ]

        answer: str = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()
