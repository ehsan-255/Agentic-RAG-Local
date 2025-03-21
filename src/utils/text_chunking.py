"""
Enhanced text chunking utilities for the RAG system.
Implements word-based chunking with overlap and code block preservation.
"""

import re
from typing import List, Tuple


def count_words(text: str) -> int:
    """
    Count the number of words in a text.
    
    Args:
        text: The text to count words in
        
    Returns:
        int: Number of words
    """
    return len(re.findall(r'\b\w+\b', text))


def extract_code_blocks(text: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Extract code blocks from text to prevent them from being split during chunking.
    Supports both HTML code blocks and markdown code blocks.
    
    Args:
        text: The text to extract code blocks from
        
    Returns:
        Tuple containing:
        - Text with code blocks replaced by placeholders
        - List of (start_pos, end_pos, code_block) tuples
    """
    # Identify code blocks and their positions
    code_blocks = []
    
    # Match HTML code blocks: <pre>, <code>, etc.
    html_code_pattern = re.compile(r'<(pre|code)[^>]*>.*?</\1>', re.DOTALL)
    
    # Match Markdown code blocks: ```...```
    md_code_pattern = re.compile(r'```(?:[\w]*\n)?.*?```', re.DOTALL)
    
    # Process HTML code blocks
    for match in html_code_pattern.finditer(text):
        start, end = match.span()
        code_blocks.append((start, end, match.group(0)))
    
    # Process Markdown code blocks
    for match in md_code_pattern.finditer(text):
        start, end = match.span()
        code_blocks.append((start, end, match.group(0)))
    
    # Sort code blocks by start position
    code_blocks.sort(key=lambda x: x[0])
    
    # Replace code blocks with placeholders
    if code_blocks:
        # Create a new string with placeholders
        result = []
        last_end = 0
        
        for start, end, block in code_blocks:
            # Add text between the last code block and this one
            result.append(text[last_end:start])
            # Add a placeholder for this code block
            result.append(f"__CODE_BLOCK_{len(result)//2}__")
            last_end = end
        
        # Add any remaining text after the last code block
        result.append(text[last_end:])
        
        return ''.join(result), code_blocks
    
    return text, []


def restore_code_blocks(chunked_text: str, code_blocks: List[Tuple[int, int, str]]) -> str:
    """
    Restore code block placeholders in chunked text.
    
    Args:
        chunked_text: Text with code block placeholders
        code_blocks: List of (start_pos, end_pos, code_block) tuples
        
    Returns:
        str: Text with code blocks restored
    """
    for i, (_, _, block) in enumerate(code_blocks):
        placeholder = f"__CODE_BLOCK_{i}__"
        chunked_text = chunked_text.replace(placeholder, block)
    
    return chunked_text


def split_text_into_chunks_with_words(
    text: str, 
    target_words_per_chunk: int = 250, 
    overlap_words: int = 50
) -> List[str]:
    """
    Split text into chunks based on a target word count, preserving code blocks and paragraph structure.
    
    Args:
        text: Text to split into chunks
        target_words_per_chunk: Target number of words per chunk
        overlap_words: Number of words to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if not text:
        return []
    
    # Check if text is shorter than target size
    if count_words(text) <= target_words_per_chunk:
        return [text]
    
    # Extract code blocks to prevent them from being split
    text_with_placeholders, code_blocks = extract_code_blocks(text)
    
    # Split into paragraphs
    paragraphs = text_with_placeholders.split("\n\n")
    
    chunks = []
    current_chunk = ""
    current_words = 0
    prev_chunk_end_words = ""  # Store the ending words of previous chunk for overlap
    
    for paragraph in paragraphs:
        paragraph_words = count_words(paragraph)
        
        # If a single paragraph exceeds the limit, we need to split it
        if paragraph_words > target_words_per_chunk:
            # If we have content already, finish the current chunk
            if current_chunk:
                chunks.append(restore_code_blocks(current_chunk, code_blocks))
                # Get the last 'overlap_words' to add to the beginning of the next chunk
                words = re.findall(r'\b\w+\b', current_chunk)
                if len(words) > overlap_words:
                    prev_chunk_end_words = " ".join(words[-overlap_words:])
                else:
                    prev_chunk_end_words = " ".join(words)
                
                current_chunk = prev_chunk_end_words + "\n\n"
                current_words = count_words(prev_chunk_end_words)
            
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence_words = count_words(sentence)
                
                # If adding this sentence would exceed the target, finish the chunk
                if current_words + sentence_words > target_words_per_chunk and current_words > 0:
                    chunks.append(restore_code_blocks(current_chunk, code_blocks))
                    
                    # Get the last 'overlap_words' for the next chunk
                    words = re.findall(r'\b\w+\b', current_chunk)
                    if len(words) > overlap_words:
                        prev_chunk_end_words = " ".join(words[-overlap_words:])
                    else:
                        prev_chunk_end_words = " ".join(words)
                    
                    current_chunk = prev_chunk_end_words + "\n\n"
                    current_words = count_words(prev_chunk_end_words)
                
                current_chunk += sentence + " "
                current_words += sentence_words
        else:
            # If adding this paragraph would exceed the target, finish the chunk
            if current_words + paragraph_words > target_words_per_chunk and current_words > 0:
                chunks.append(restore_code_blocks(current_chunk, code_blocks))
                
                # Get the last 'overlap_words' for the next chunk
                words = re.findall(r'\b\w+\b', current_chunk)
                if len(words) > overlap_words:
                    prev_chunk_end_words = " ".join(words[-overlap_words:])
                else:
                    prev_chunk_end_words = " ".join(words)
                
                current_chunk = prev_chunk_end_words + "\n\n"
                current_words = count_words(prev_chunk_end_words)
            
            current_chunk += paragraph + "\n\n"
            current_words += paragraph_words
    
    # Add any remaining content as the last chunk
    if current_chunk:
        chunks.append(restore_code_blocks(current_chunk, code_blocks))
    
    return chunks


def enhanced_chunk_text(
    text: str, 
    chunk_size_words: int = 250, 
    overlap_words: int = 50
) -> List[str]:
    """
    Main function to split text into chunks with word-based size and overlap.
    
    Args:
        text: Text to split into chunks
        chunk_size_words: Target word count per chunk
        overlap_words: Word overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Log the word-based chunking parameters
    total_words = count_words(text)
    overlap_percentage = (overlap_words / chunk_size_words) * 100 if chunk_size_words > 0 else 0
    
    print(f"Word-based chunking: {total_words} total words, {chunk_size_words} words per chunk, {overlap_words} words overlap ({overlap_percentage:.1f}%)")
    
    return split_text_into_chunks_with_words(text, chunk_size_words, overlap_words)


# Legacy character-based chunking function for backward compatibility
def character_based_chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Legacy character-based chunking method (kept for backward compatibility).
    
    Args:
        text: Text to split into chunks
        max_chars: Maximum characters per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Log the character-based chunking parameters
    print(f"Character-based chunking: {len(text)} total characters, {max_chars} characters per chunk")
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by double newlines to maintain paragraph structure
    paragraphs = text.split("\n\n")
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds the chunk size and we already have content,
        # finish the current chunk and start a new one
        if len(current_chunk) + len(paragraph) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        
        # If a single paragraph is larger than the chunk size, split it by sentences
        if len(paragraph) > max_chars:
            # Simple sentence splitting (not perfect but functional)
            sentences = paragraph.replace(". ", ".|").replace("? ", "?|").replace("! ", "!|").split("|")
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
        else:
            current_chunk += paragraph + "\n\n"
    
    # Add any remaining content
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks 