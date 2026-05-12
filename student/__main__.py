from student.indexing import Indexing
from student.generation import AnswerGenerator


def main() -> None:
    """
    """
    index = Indexing()
    index.load_files("data/vllm-0.10.1/")
    index.build_chunks()
    index.build_indexes()
    chunks = index.retrieve("Hello world", 1)

    agent = AnswerGenerator()
    print(agent.generate_answer("Hello world?", chunks))


if __name__ == "__main__":
    main()
