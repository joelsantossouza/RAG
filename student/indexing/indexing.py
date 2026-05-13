from pathlib import Path, PosixPath
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from bm25s import BM25
from bm25s.tokenization import Tokenized as BM25Tokenized
import bm25s
from ..models import (
    MinimalSource,
    UnansweredQuestion,
    MinimalSearchResults
)


class Indexing:
    """
    """

    def __init__(self) -> None:
        self.files_path: list[PosixPath] = []
        self.chunks: list[MinimalSource] = []
        self.corpus: list[str] = []
        self.bm25: BM25 = BM25()

    def load_files(self, path: str) -> None:
        root: Path = Path(path)

        if root.is_file():
            self.files_path.append(root)
            return

        if not root.is_dir():
            return

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            self.files_path.append(file_path)

    def build_chunks(self, chunk_size: int = 2000) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            keep_separator=True,
            add_start_index=True,
        )

        for file_path in self.files_path:
            content: str = file_path.read_text(
                encoding="utf-8", errors="ignore"
            )
            documents: list[Document] = splitter.create_documents([content])

            for doc in documents:
                start: int = doc.metadata["start_index"]
                end: int = start + len(doc.page_content)

                self.chunks.append(
                    MinimalSource(
                        file_path=file_path.as_posix(),
                        first_character_index=start,
                        last_character_index=end,
                        data=content[start:end]
                    )
                )

    def build_indexes(self) -> None:
        self.corpus = [
            chunk.data for chunk in self.chunks
        ]
        tokenized_corpus: BM25Tokenized = bm25s.tokenize(self.corpus)
        self.bm25.index(tokenized_corpus)

    def retrieve_batch(self, questions: list[UnansweredQuestion],
                       top_k: int = 5) -> list[MinimalSearchResults]:
        tokenized_queries: BM25Tokenized = bm25s.tokenize(
            [q.question for q in questions]
        )
        chunks_ids, _ = self.bm25.retrieve(
            tokenized_queries,
            k=top_k,
        )
        return [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[self.chunks[i] for i in chunks_ids[q]]
            )
            for q, question in enumerate(questions)
        ]
