import os
import json
import zstandard as zstd
from typing import Iterator, Tuple, Optional, Any
import io

class Reader:
    """Reader for OpenWebText2 format using zstandard compression."""
    
    def __init__(self):
        self.dctx = zstd.ZstdDecompressor()
    
    def _read_zst(self, path: str) -> Iterator[str]:
        """Read a zstandard compressed file line by line."""
        with open(path, 'rb') as fh:
            with self.dctx.stream_reader(fh) as reader:
                # Create a buffered reader for efficient line reading
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    yield line.strip()
    
    def read_jsonl(self, path: str, get_meta: bool = True) -> Iterator[Tuple[str, Optional[dict]]]:
        """Read a jsonl.zst file and yield (text, metadata) tuples."""
        for line in self._read_zst(path):
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get('text', '').strip()
                meta = {k: v for k, v in obj.items() if k != 'text'} if get_meta else None
                if text:  # Only yield if there's actual text content
                    yield text, meta
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error processing line in {path}: {e}")
                continue 