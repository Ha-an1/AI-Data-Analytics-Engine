import pandas as pd
import time
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

class StreamSimulator:
    def __init__(self, chunk_size=1000, delay_seconds=1.0):
        self.chunk_size = chunk_size
        self.delay_seconds = delay_seconds

    def simulate_stream(self, filepath: str) -> Iterator[pd.DataFrame]:
        """
        Simulates a real-time data stream by chunking a static dataset.
        Yields pandas DataFrames to represent micro-batches.
        """
        try:
            # Use chunksize to avoid loading massive files entirely
            for chunk in pd.read_csv(filepath, chunksize=self.chunk_size):
                logger.info(f"Emitting micro-batch chunk of {len(chunk)} rows...")
                yield chunk
                time.sleep(self.delay_seconds)
        except Exception as e:
            logger.error(f"Stream simulation failed: {e}")
