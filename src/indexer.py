from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import Language # Consider using this splitter by using the source code file extention to determine the language
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import shutil
from constants import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL

def indexer(code_path: Path):
    # ====================================
    # LOAD CODE
    # ====================================

    code_text = code_path.read_text()

    if CHROMA_DIR.exists():
        for item in CHROMA_DIR.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # ====================================
    # SPLITTER
    # ====================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # language_f = get_language(code_path)
    # text_splitter = RecursiveCharacterTextSplitter.from_language(
    #     language=Language.language_f,
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP
    # )

    texts = text_splitter.split_text(code_text) 

    # ====================================
    # EMBEDDINGS
    # ====================================

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL
    )

    # ====================================
    # CHROMA DB
    # ====================================

    # IMPORTANTE:
    # Essa etapa pode demorar bastante
    # em textos grandes.
    #
    # Ela deve ser executada apenas
    # quando o código mudar.
    #

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )