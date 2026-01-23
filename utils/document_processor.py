"""
Document Processor for Lenny's Podcast Transcripts

Handles loading MD files, parsing YAML frontmatter, and chunking text
with metadata preservation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import frontmatter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default paths
TRANSCRIPTS_DIR = Path(__file__).parent.parent / "data" / "transcripts" / "episodes"


def discover_transcripts(transcripts_dir: Path = TRANSCRIPTS_DIR) -> List[Path]:
    """
    Discover all transcript MD files in the transcripts directory.

    Args:
        transcripts_dir: Path to the episodes directory

    Returns:
        List of paths to transcript.md files
    """
    transcript_files = []

    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_dir}")

    for guest_dir in transcripts_dir.iterdir():
        if guest_dir.is_dir():
            transcript_path = guest_dir / "transcript.md"
            if transcript_path.exists():
                transcript_files.append(transcript_path)

    return sorted(transcript_files)


def load_transcript(file_path: Path) -> str:
    """
    Load the raw content of a transcript file.

    Args:
        file_path: Path to the transcript.md file

    Returns:
        Raw file content as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_frontmatter(content: str) -> Dict[str, Any]:
    """
    Parse YAML frontmatter from transcript content.

    Args:
        content: Raw transcript content with YAML frontmatter

    Returns:
        Dictionary of metadata (guest, title, date, etc.)
    """
    post = frontmatter.loads(content)
    return dict(post.metadata)


def get_transcript_body(content: str) -> str:
    """
    Extract the transcript body (text after frontmatter).

    Args:
        content: Raw transcript content with YAML frontmatter

    Returns:
        Transcript text without frontmatter
    """
    post = frontmatter.loads(content)
    return post.content


def chunk_transcript(
    body: str,
    metadata: Dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Split transcript body into chunks with metadata attached.

    Args:
        body: Transcript text
        metadata: Metadata dict from frontmatter
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(body)

    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_metadata = {
            **metadata,
            'chunk_id': i,
            'chunk_total': len(chunks)
        }
        result.append({
            'text': chunk_text,
            'metadata': chunk_metadata
        })

    return result


def process_transcript(file_path: Path) -> List[Dict[str, Any]]:
    """
    Process a single transcript file: load, parse, and chunk.

    Args:
        file_path: Path to transcript.md file

    Returns:
        List of chunks with metadata
    """
    content = load_transcript(file_path)
    metadata = parse_frontmatter(content)
    body = get_transcript_body(content)

    # Add source file info to metadata
    metadata['source_file'] = str(file_path)
    metadata['guest_slug'] = file_path.parent.name

    return chunk_transcript(body, metadata)


def process_all_transcripts(
    transcripts_dir: Path = TRANSCRIPTS_DIR,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Process all transcripts and return all chunks.

    Args:
        transcripts_dir: Path to episodes directory
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of all chunks from all transcripts
    """
    transcript_files = discover_transcripts(transcripts_dir)
    all_chunks = []

    for i, file_path in enumerate(transcript_files):
        try:
            chunks = process_transcript(file_path)
            all_chunks.extend(chunks)

            if progress_callback:
                progress_callback(i + 1, len(transcript_files))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return all_chunks


# Utility function for testing
def get_transcript_count(transcripts_dir: Path = TRANSCRIPTS_DIR) -> int:
    """Get the number of available transcripts."""
    return len(discover_transcripts(transcripts_dir))


if __name__ == "__main__":
    # Quick test
    print(f"Found {get_transcript_count()} transcripts")

    transcripts = discover_transcripts()
    if transcripts:
        # Test with first transcript
        test_file = transcripts[0]
        print(f"\nTesting with: {test_file}")

        chunks = process_transcript(test_file)
        print(f"Created {len(chunks)} chunks")

        if chunks:
            print(f"\nFirst chunk metadata: {chunks[0]['metadata'].get('title', 'N/A')}")
            print(f"First chunk text preview: {chunks[0]['text'][:200]}...")
