import os
import json
from llama_cpp import Llama
from tqdm import tqdm
from ..models import (
    MinimalSource,
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    RagDataset,
)

MODEL_PATH = "qwen3-0.6b-q4_k_m.gguf"


class AnswerGenerator:
    """"""

    def __init__(self) -> None:
        self.model = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_threads=os.cpu_count() or 8,
            n_gpu_layers=0,
            verbose=False
        )

    def generate_answer(
        self,
        questions: list[MinimalSearchResults],
        max_tokens: int = 100
    ) -> list[MinimalAnswer]:
        answers: list[MinimalAnswer] = []

        for search_result in tqdm(
            questions,
            total=len(questions),
            desc="Generating answers",
            leave=True,
            colour="yellow"
        ):
            context = "\n\n---\n\n".join(
                chunk.data[:100]
                for chunk in search_result.retrieved_sources
            )
            prompt = f"""<|im_start|>system
Answer using ONLY the provided context. If the answer is not in the context, say so.<|im_end|>
<|im_start|>user /no_think
Context:
{context}

Question:
{search_result.question}<|im_end|>
<|im_start|>assistant
"""
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                echo=False
            )
            answer: str = output["choices"][0]["text"].strip()

            answers.append(
                MinimalAnswer(
                    question_id=search_result.question_id,
                    question=search_result.question,
                    retrieved_sources=search_result.retrieved_sources,
                    answer=answer
                )
            )

        return answers


class DataSet:

    def __init__(self) -> None:
        ...

    def extract_questions(self, path: str) -> list[UnansweredQuestion]:
        with open(path, "r", encoding="utf-8") as fd:
            data = json.load(fd)
        dataset = RagDataset(**data)
        questions: list[UnansweredQuestion] = [
            UnansweredQuestion(
                question_id=q.question_id,
                question=q.question
            )
            for q in dataset.rag_questions
        ]
        return questions

    def save_search_results(
        self,
        search_results: list[MinimalSearchResults],
        k: int,
        output_path: str
    ) -> None:
        output = StudentSearchResults(
            search_results=search_results,
            k=k
        )
        self._write(output.model_dump(
            exclude={"search_results": {"__all__": {"content"}}}
        ), output_path)

    def save_answers(
        self,
        answers: list[MinimalAnswer],
        k: int,
        output_path: str
    ) -> None:
        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=k
        )
        self._write(output.model_dump(
            exclude={"search_results": {"__all__": {"content"}}}
        ), output_path)

    def save_sources(
        self,
        sources: list[MinimalSource],
        output_path: str
    ) -> None:
        data = [
            {"source": source.model_dump()}
            for source in sources
        ]
        self._write(data, output_path)

    def _write(self, data: dict, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fd:
            json.dump(data, fd, indent=4, ensure_ascii=False)
