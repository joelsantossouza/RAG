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
            dtype=torch.float16 if self.device.type in ("cuda", "mps")
            else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

    def generate_answer(self, question: str, resource: list[Chunk],
                        max_tokens: int = 100) -> str:
        context: str = "\n\n".join(chunk.data for chunk in resource)
        prompt: str = (
            "You are a helpful assistant.\n\n"
            "Answer using ONLY the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][
            inputs.input_ids.shape[1]:
        ]
        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        return answer.strip()
