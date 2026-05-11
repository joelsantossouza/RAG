from pathlib import Path, PosixPath
from ..models import MinimalSource
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



class Indexing:
    """
    """

    def __init__(self) -> None:
        self.files_path: list[PosixPath] = []
        self.chunks: list[MinimalSource] = []

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
            chunk_overlap=0,
            keep_separator=False,
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
                    )
                )
