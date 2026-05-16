import fire
import os
from student.indexing import Indexing
from student.generation import AnswerGenerator, DataSet
from student.models import UnansweredQuestion

REPOSITORY_PATH: str = "data/raw/vllm-0.10.1"
PROCESSED_CHUNKS_PATH: str = "data/processed/chunks/metadata.json"
QUESTIONS_PATH: str = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
RESULT_DIR: str = "data/output/search_results"


class StudentCli:
    """"""

    def __init__(self) -> None:
        self.indexer: Indexing = Indexing()
        self.dataset: DataSet = DataSet()
        self.answer_gen: AnswerGenerator = AnswerGenerator()

    def index(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size <= 0:
            print("Error: max_chunk_size must be a positive integer.")
            exit(1)

        print(f"Loading files under {REPOSITORY_PATH}...")
        self.indexer.load_files(REPOSITORY_PATH)

        print(f"Building chunks {REPOSITORY_PATH}...")
        self.indexer.build_chunks(max_chunk_size)

        print(f"Saving chunks on {PROCESSED_CHUNKS_PATH}...")
        self.dataset.save_sources(
            self.indexer.chunks, PROCESSED_CHUNKS_PATH
        )

        print("Building indexes...")
        self.indexer.build_indexes()

    def answer(self, question: str, k: int = 10) -> None:
        if not question.strip():
            print("Question cannot be empty")
            exit(1)

        question_structured = UnansweredQuestion(
            question=question
        )
        print("Searching for context...")
        search_results: list = self.indexer.retrieve_batch(
            [question_structured], k
        )

        print(f"Asking: {question}...")
        answers: list = self.answer_gen.generate_answer(search_results)
        print(f"Got response: {answers[0].answer}")

    def search_dataset(self, dataset_path: str = QUESTIONS_PATH,
                       k: int = 10,
                       save_directory: str = RESULT_DIR) -> None:
        print("Extracting questions...")
        questions = self.dataset.extract_questions(dataset_path)

        print("Searching for context...")
        search_results: list = self.indexer.retrieve_batch(questions, k)

        file_name: str = os.path.basename(dataset_path)
        result_path: str = os.path.join(save_directory, file_name)

        os.makedirs(save_directory, exist_ok=True)
        self.dataset.save_search_results(search_results, k, result_path)
        print(f"Saved student_search_results to {result_path}")


if __name__ == "__main__":
    try:
        fire.Fire(StudentCli)
    except Exception as error_msg:
        print(f"Error: {error_msg}")
        exit(1)
