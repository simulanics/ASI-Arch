class Config:
    """Configuration settings for the experiment."""
    # Target file
    SOURCE_FILE: str = "evolve file"
    
    # Training script
    BASH_SCRIPT: str = "your training script"
    
    # Experiment results
    RESULT_FILE: str = "./files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"
    
    # Debug file
    DEBUG_FILE: str = "./files/debug/training_error.txt"
    
    # Code pool directory
    CODE_POOL: str = "./pool"
    
    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 3
    
    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 10
    
    # RAG service URL
    RAG: str = "http://172.29.57.33:13142/"
    
    # Database URL
    DATABASE: str = "http://172.29.57.33:13142/"