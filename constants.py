from pathlib import Path

# Paths

SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_DIR: Path = SCRIPT_DIR.parent
CONFIG_PATH: Path = SCRIPT_DIR / "config.json"
HEURISTICS_PATH: Path = SCRIPT_DIR / "data" / "heuristics.json"
CHECKPOINT_PATH: Path = SCRIPT_DIR / "data" / "checkpoint.json"
LOG_PATH: Path = SCRIPT_DIR / "data" / "log.json"
STAGES: list[str] = [
    "static_analysis",
    "eval_loop",
    "initial_refactoring",
    "refactoring_loop",
    "saving",
]
CHROMA_DIR: Path = SCRIPT_DIR / "chroma_db"

# Maximum numbers of times the evaluator/generator will try to generate a response to be judged.
MAX_TENTATIVES: int = 3

# Maximum number of concurrent retrievals
MAX_ASYNC_WORKERS = 1  # effectly eliminating async to facilitate iteration on the code, increase latter if retriever code allows

# Chunking Config, a bigger the chunck is faster and less precise, while a smaller chunck is more precise and slower
CHUNK_SIZE = 1500

# Chunks overlaps, helps sustain context
CHUNK_OVERLAP = 150

# Embeddings models
EMBED_MODEL = "nomic-embed-text"

# Amount of retrieved chunks
TOP_K = 3

# Temperature, the higher the value the more creative the llm will be
TEMPERATURE = 0.5

# Maximum size of the response in tokens
MAX_TOKENS = 4096
