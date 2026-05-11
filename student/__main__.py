from student.indexing import Indexing

def main() -> None:
    """
    """
    index = Indexing()
    index.load_files("/home/joesanto/Downloads/10kb.txt")
    index.build_chunks()
    for chunk in index.chunks:
        print(chunk)
        print()

if __name__ == "__main__":
    main()
