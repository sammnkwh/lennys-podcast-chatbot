"""
Unit tests for Document Processor (Component 1)
"""

import pytest
from pathlib import Path
from utils.document_processor import (
    discover_transcripts,
    load_transcript,
    parse_frontmatter,
    get_transcript_body,
    chunk_transcript,
    process_transcript,
    process_all_transcripts,
    get_transcript_count,
    TRANSCRIPTS_DIR
)


# Sample content for testing
SAMPLE_TRANSCRIPT = """---
guest: Test Guest
title: Test Episode Title
youtube_url: https://www.youtube.com/watch?v=abc123
publish_date: 2024-01-15
description: A test episode description.
duration: '45:00'
keywords:
- test
- example
---

This is the transcript body.

It has multiple paragraphs.

And some more content here that we can use for testing the chunking functionality.
"""


class TestDiscoverTranscripts:
    """Tests for discover_transcripts() function"""

    def test_discovers_transcript_files(self):
        """Should find MD files in the transcripts directory"""
        transcripts = discover_transcripts()
        assert len(transcripts) > 0, "Should find at least one transcript"

    def test_returns_path_objects(self):
        """Should return Path objects"""
        transcripts = discover_transcripts()
        assert all(isinstance(t, Path) for t in transcripts)

    def test_all_files_are_transcript_md(self):
        """All returned files should be named transcript.md"""
        transcripts = discover_transcripts()
        assert all(t.name == "transcript.md" for t in transcripts)

    def test_raises_error_for_nonexistent_dir(self):
        """Should raise FileNotFoundError for nonexistent directory"""
        with pytest.raises(FileNotFoundError):
            discover_transcripts(Path("/nonexistent/path"))


class TestLoadTranscript:
    """Tests for load_transcript() function"""

    def test_loads_file_content(self):
        """Should load raw file content"""
        transcripts = discover_transcripts()
        if transcripts:
            content = load_transcript(transcripts[0])
            assert isinstance(content, str)
            assert len(content) > 0

    def test_content_starts_with_frontmatter(self):
        """Loaded content should start with YAML frontmatter delimiter"""
        transcripts = discover_transcripts()
        if transcripts:
            content = load_transcript(transcripts[0])
            assert content.startswith("---")


class TestParseFrontmatter:
    """Tests for parse_frontmatter() function"""

    def test_extracts_metadata_dict(self):
        """Should return a dictionary of metadata"""
        metadata = parse_frontmatter(SAMPLE_TRANSCRIPT)
        assert isinstance(metadata, dict)

    def test_extracts_guest(self):
        """Should extract guest name"""
        metadata = parse_frontmatter(SAMPLE_TRANSCRIPT)
        assert metadata.get("guest") == "Test Guest"

    def test_extracts_title(self):
        """Should extract episode title"""
        metadata = parse_frontmatter(SAMPLE_TRANSCRIPT)
        assert metadata.get("title") == "Test Episode Title"

    def test_extracts_publish_date(self):
        """Should extract publish date"""
        metadata = parse_frontmatter(SAMPLE_TRANSCRIPT)
        # frontmatter library may parse as date object or string
        assert "publish_date" in metadata

    def test_extracts_keywords_as_list(self):
        """Should extract keywords as a list"""
        metadata = parse_frontmatter(SAMPLE_TRANSCRIPT)
        assert isinstance(metadata.get("keywords"), list)
        assert "test" in metadata["keywords"]

    def test_handles_real_transcript(self):
        """Should parse real transcript frontmatter"""
        transcripts = discover_transcripts()
        if transcripts:
            content = load_transcript(transcripts[0])
            metadata = parse_frontmatter(content)
            assert "guest" in metadata
            assert "title" in metadata


class TestGetTranscriptBody:
    """Tests for get_transcript_body() function"""

    def test_returns_body_without_frontmatter(self):
        """Should return content without YAML frontmatter"""
        body = get_transcript_body(SAMPLE_TRANSCRIPT)
        assert "---" not in body
        assert "guest:" not in body

    def test_body_contains_transcript_text(self):
        """Should contain the actual transcript text"""
        body = get_transcript_body(SAMPLE_TRANSCRIPT)
        assert "This is the transcript body" in body

    def test_body_is_string(self):
        """Should return a string"""
        body = get_transcript_body(SAMPLE_TRANSCRIPT)
        assert isinstance(body, str)


class TestChunkTranscript:
    """Tests for chunk_transcript() function"""

    def test_returns_list_of_chunks(self):
        """Should return a list"""
        body = "This is some text. " * 100  # Create enough text to chunk
        metadata = {"guest": "Test", "title": "Test Title"}
        chunks = chunk_transcript(body, metadata)
        assert isinstance(chunks, list)

    def test_each_chunk_has_text_and_metadata(self):
        """Each chunk should have 'text' and 'metadata' keys"""
        body = "This is some text. " * 100
        metadata = {"guest": "Test", "title": "Test Title"}
        chunks = chunk_transcript(body, metadata)
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk

    def test_metadata_preserved_in_chunks(self):
        """Original metadata should be preserved in each chunk"""
        body = "This is some text. " * 100
        metadata = {"guest": "Test Guest", "title": "Test Title"}
        chunks = chunk_transcript(body, metadata)
        for chunk in chunks:
            assert chunk["metadata"]["guest"] == "Test Guest"
            assert chunk["metadata"]["title"] == "Test Title"

    def test_chunk_id_added(self):
        """Each chunk should have a chunk_id"""
        body = "This is some text. " * 100
        metadata = {"guest": "Test", "title": "Test Title"}
        chunks = chunk_transcript(body, metadata)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_id"] == i

    def test_chunk_total_added(self):
        """Each chunk should have chunk_total"""
        body = "This is some text. " * 100
        metadata = {"guest": "Test", "title": "Test Title"}
        chunks = chunk_transcript(body, metadata)
        total = len(chunks)
        for chunk in chunks:
            assert chunk["metadata"]["chunk_total"] == total

    def test_respects_chunk_size(self):
        """Chunks should respect the specified size (approximately)"""
        body = "This is some text. " * 200
        metadata = {"guest": "Test"}
        chunks = chunk_transcript(body, metadata, chunk_size=500, chunk_overlap=50)
        # Allow some flexibility due to splitter behavior
        for chunk in chunks:
            assert len(chunk["text"]) <= 600  # chunk_size + some buffer


class TestProcessTranscript:
    """Tests for process_transcript() function"""

    def test_returns_chunks_with_metadata(self):
        """Should return list of chunks from a transcript file"""
        transcripts = discover_transcripts()
        if transcripts:
            chunks = process_transcript(transcripts[0])
            assert len(chunks) > 0
            assert "text" in chunks[0]
            assert "metadata" in chunks[0]

    def test_adds_source_file_to_metadata(self):
        """Should add source_file to metadata"""
        transcripts = discover_transcripts()
        if transcripts:
            chunks = process_transcript(transcripts[0])
            assert "source_file" in chunks[0]["metadata"]

    def test_adds_guest_slug_to_metadata(self):
        """Should add guest_slug (directory name) to metadata"""
        transcripts = discover_transcripts()
        if transcripts:
            chunks = process_transcript(transcripts[0])
            assert "guest_slug" in chunks[0]["metadata"]


class TestProcessAllTranscripts:
    """Tests for process_all_transcripts() function"""

    def test_returns_chunks_from_multiple_transcripts(self):
        """Should return chunks from all transcripts"""
        # Limit to first few for speed
        transcripts = discover_transcripts()[:3]
        if transcripts:
            all_chunks = []
            for t in transcripts:
                all_chunks.extend(process_transcript(t))
            assert len(all_chunks) > 0

    def test_progress_callback_called(self):
        """Progress callback should be called for each transcript"""
        call_count = [0]

        def callback(current, total):
            call_count[0] += 1

        # Process just a few transcripts
        transcripts_dir = TRANSCRIPTS_DIR
        transcripts = discover_transcripts(transcripts_dir)[:2]

        if len(transcripts) >= 2:
            # We need to test with actual transcripts
            process_all_transcripts(transcripts_dir, progress_callback=callback)
            assert call_count[0] > 0


class TestGetTranscriptCount:
    """Tests for get_transcript_count() function"""

    def test_returns_positive_integer(self):
        """Should return a positive integer"""
        count = get_transcript_count()
        assert isinstance(count, int)
        assert count > 0

    def test_matches_discover_length(self):
        """Should match length of discover_transcripts()"""
        count = get_transcript_count()
        transcripts = discover_transcripts()
        assert count == len(transcripts)
