"""
LLM Chain for Lenny's Podcast RAG Chatbot

Handles Gemini LLM setup, RAG chain with conversation memory,
response parsing, and citation formatting.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048

# Default system prompt for Lenny's Podcast Q&A
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Lenny's Podcast episodes.
You have access to transcripts from the podcast and should provide informative, accurate answers based on the content.

Guidelines:
- Answer questions based on the provided context from podcast transcripts
- Always cite your sources using the format: From "[Episode Title]" with [Guest Name] ([Date])
- If the context doesn't contain relevant information, say so honestly
- Be conversational and helpful
- Keep answers concise but comprehensive

After your answer, provide 3-5 follow-up questions that the user might want to explore next.
Format the follow-up questions in a section starting with "FOLLOW-UP QUESTIONS:" with each question on a new line starting with "- ".
"""


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> ChatGoogleGenerativeAI:
    """
    Get Gemini LLM instance.

    Args:
        model: Model name (gemini-1.5-flash or gemini-2.0-flash)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum output tokens

    Returns:
        ChatGoogleGenerativeAI instance

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens
    )


def format_context(search_results: List[Dict[str, Any]]) -> str:
    """
    Format search results into context string with source citations.

    Args:
        search_results: List of search results with 'text' and 'metadata'

    Returns:
        Formatted context string
    """
    if not search_results:
        return "No relevant context found."

    context_parts = []

    for i, result in enumerate(search_results, 1):
        text = result.get("text", "")
        metadata = result.get("metadata", {})

        title = metadata.get("title", "Unknown Episode")
        guest = metadata.get("guest", "Unknown Guest")
        date = metadata.get("publish_date", "Unknown Date")

        # Format the source citation
        source = f'From "{title}" with {guest} ({date})'

        context_parts.append(f"[Source {i}] {source}\n{text}")

    return "\n\n---\n\n".join(context_parts)


def format_chat_history(messages: List[Dict[str, str]]) -> List[Any]:
    """
    Convert chat history to LangChain message format.

    Args:
        messages: List of dicts with 'role' and 'content'

    Returns:
        List of LangChain message objects
    """
    langchain_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))

    return langchain_messages


def build_prompt(
    query: str,
    context: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[Any]:
    """
    Build the full prompt with system message, context, history, and query.

    Args:
        query: User's question
        context: Formatted context from search results
        system_prompt: System prompt for the assistant
        chat_history: Optional list of previous messages

    Returns:
        List of LangChain messages
    """
    messages = []

    # System message with context
    full_system = f"""{system_prompt}

CONTEXT FROM PODCAST TRANSCRIPTS:
{context}
"""
    messages.append(SystemMessage(content=full_system))

    # Add chat history if provided
    if chat_history:
        messages.extend(format_chat_history(chat_history))

    # Add current query
    messages.append(HumanMessage(content=query))

    return messages


def parse_response(llm_output: str) -> Dict[str, Any]:
    """
    Parse LLM output to separate answer from follow-up questions.

    Args:
        llm_output: Raw LLM response text

    Returns:
        Dict with 'answer' (str) and 'followups' (list[str])
    """
    # Try to find the follow-up questions section
    followup_patterns = [
        r"FOLLOW-UP QUESTIONS?:",
        r"Follow-up Questions?:",
        r"Follow-Up Questions?:",
        r"Suggested Questions?:",
        r"You might also ask:",
        r"Related questions?:",
    ]

    answer = llm_output
    followups = []

    for pattern in followup_patterns:
        match = re.search(pattern, llm_output, re.IGNORECASE)
        if match:
            # Split at the pattern
            answer = llm_output[:match.start()].strip()
            followup_section = llm_output[match.end():].strip()

            # Extract individual questions
            # Look for lines starting with -, *, or numbers
            question_pattern = r'(?:^|\n)\s*(?:[-*•]|\d+[.):])\s*(.+?)(?=\n\s*(?:[-*•]|\d+[.):])|$)'
            matches = re.findall(question_pattern, followup_section, re.MULTILINE | re.DOTALL)

            for q in matches:
                q = q.strip()
                # Clean up the question
                q = re.sub(r'\s+', ' ', q)
                if q and len(q) > 10:  # Filter out too-short strings
                    # Ensure it ends with a question mark
                    if not q.endswith('?'):
                        q = q.rstrip('.') + '?'
                    followups.append(q)

            break

    # If no follow-ups found using patterns, try simple line-based extraction
    if not followups:
        lines = llm_output.split('\n')
        in_followup_section = False

        for line in lines:
            line = line.strip()

            # Check if we're entering a follow-up section
            if any(re.search(p, line, re.IGNORECASE) for p in followup_patterns):
                in_followup_section = True
                continue

            if in_followup_section and line:
                # Remove bullet points or numbers
                cleaned = re.sub(r'^[-*•\d.):]+\s*', '', line).strip()
                if cleaned and len(cleaned) > 10 and '?' in cleaned:
                    followups.append(cleaned)

    # Limit to 5 follow-ups
    followups = followups[:5]

    return {
        "answer": answer,
        "followups": followups
    }


def generate_response(
    query: str,
    search_results: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE
) -> Dict[str, Any]:
    """
    Generate a response using the RAG chain.

    Args:
        query: User's question
        search_results: Results from vector search
        chat_history: Optional conversation history
        system_prompt: System prompt to use
        model: LLM model name
        temperature: Sampling temperature

    Returns:
        Dict with 'answer', 'followups', and 'raw_response'
    """
    # Format context from search results
    context = format_context(search_results)

    # Build the prompt
    messages = build_prompt(
        query=query,
        context=context,
        system_prompt=system_prompt,
        chat_history=chat_history
    )

    # Get LLM and generate response
    llm = get_llm(model=model, temperature=temperature)
    response = llm.invoke(messages)

    # Parse the response
    raw_output = response.content
    parsed = parse_response(raw_output)

    return {
        "answer": parsed["answer"],
        "followups": parsed["followups"],
        "raw_response": raw_output
    }


def test_llm_connection(model: str = DEFAULT_MODEL) -> bool:
    """
    Test that we can connect to the LLM.

    Args:
        model: Model to test

    Returns:
        True if connection successful
    """
    try:
        llm = get_llm(model=model)
        response = llm.invoke([HumanMessage(content="Say 'Hello' in one word.")])
        return bool(response.content)
    except Exception as e:
        print(f"LLM connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing LLM chain...")

    # Test connection
    print("\n1. Testing LLM connection...")
    if test_llm_connection():
        print("✓ LLM connection successful")
    else:
        print("✗ LLM connection failed")

    # Test response parsing
    print("\n2. Testing response parsing...")
    sample_response = """Based on the podcast transcripts, product-market fit is when...

This was discussed in detail by several guests.

From "Finding Product-Market Fit" with Marc Andreessen (2023-05-15):
The key insight is that you know you have PMF when customers are pulling the product from you.

FOLLOW-UP QUESTIONS:
- How do you measure product-market fit quantitatively?
- What are common signs that you don't have product-market fit?
- How long does it typically take to achieve product-market fit?
"""

    parsed = parse_response(sample_response)
    print(f"✓ Answer length: {len(parsed['answer'])} chars")
    print(f"✓ Follow-ups found: {len(parsed['followups'])}")
    for q in parsed['followups']:
        print(f"  - {q}")
