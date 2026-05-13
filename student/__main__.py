from student.indexing import Indexing
from student.generation import AnswerGenerator, DataSet


def main() -> None:
    """
    """
    index: Indexing = Indexing()
    index.load_files("data/vllm-0.10.1/")
    index.build_chunks()
    index.build_indexes()

    dataset: DataSet = DataSet()
    questions = dataset.extract_questions(
        "data/datasets_public/public/UnansweredQuestions/dataset_code_public.json"
    )

    search_result = index.retrieve_batch(questions, 1)

    agent = AnswerGenerator()
    answers = agent.generate_answer(search_result)
    print(answers)

if __name__ == "__main__":
    main()
