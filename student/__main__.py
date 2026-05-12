from student.indexing import Indexing
from student.generation import AnswerGenerator


def main() -> None:
    """
    """
    index = Indexing()
    index.load_files("data/vllm-0.10.1/")
    index.build_chunks()
    index.build_indexes()
    question = "What activation formats does the fused batched MoE layer return in vLLM?"
    chunks = index.retrieve(question, 5)

    agent = AnswerGenerator()
    print(agent.generate_answer(question, chunks))


if __name__ == "__main__":
    main()
