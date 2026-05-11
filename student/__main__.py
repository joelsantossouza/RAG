from student.indexing import Indexing

def main() -> None:
    """
    """
    index = Indexing()
    index.load_files("data/vllm-0.10.1/")
    index.build_chunks()
    index.build_indexes()
    for chunk in index.retrieve("What activation formats does the fused batched MoE layer return in vLLM?", 1):
        print(chunk.file_path)
        print(index._read_chunk(chunk))

if __name__ == "__main__":
    main()
