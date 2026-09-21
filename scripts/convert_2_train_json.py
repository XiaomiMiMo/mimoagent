#!/usr/bin/env python3
"""
Extract messages field from JSONL file and save to specified file.
"""

import argparse
import json
from pathlib import Path

# from typing import list


def validate_and_normalize_message(message):
    """Validate and normalize a single message."""
    if not isinstance(message, dict):
        return None

    # Required fields must exist and be strings
    normalized = {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}

    # Drop empty messages
    if not normalized["role"] or not normalized["content"]:
        return None

    return normalized


def process_message_content(message):
    """
    Process message content: if content is a list, extract text from first element.

    Args:
        message: A message dict that may have a 'content' field

    Returns:
        message: Processed message with content as string
    """
    if "content" in message and isinstance(message["content"], list):
        if len(message["content"]) > 0 and isinstance(message["content"][0], dict) and "text" in message["content"][0]:
            message = message.copy()  # Make a copy to avoid modifying original
            message["content"] = message["content"][0]["text"]
    return validate_and_normalize_message(message)


def extract_messages_from_jsonl(input_file: Path, output_file: Path) -> None:
    """
    Extract messages field from each line in a JSONL file and save to a JSON file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSON file
    """
    messages_list = []

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read JSONL file line by line
    with input_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse JSON from each line
                data = json.loads(line)

                # Extract messages field if it exists
                if "messages" in data:
                    # Process each message to handle content field
                    processed_messages = []
                    for msg in data["messages"]:
                        processed_msg = process_message_content(msg)
                        processed_messages.append(processed_msg)

                    # Drop the last message if its role is "user"
                    if processed_messages and processed_messages[-1] and processed_messages[-1].get("role") == "user":
                        processed_messages.pop()

                    # Only add to messages_list if there are still messages left
                    if processed_messages:
                        messages_list.append({"messages": processed_messages})
                else:
                    print(f"Warning: Line {line_num} does not contain 'messages' field")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue

    # Save extracted messages to output file
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(messages_list, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(messages_list)} messages from {input_file}")
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract messages field from JSONL file and save to specified file")

    parser.add_argument("-i", "--input_file", type=Path, help="Path to input JSONL file")

    parser.add_argument("-o", "--output_file", type=Path, help="Path to output JSON file")

    args = parser.parse_args()

    # Validate input file exists
    if not args.input_file.exists():
        print(f"Error: Input file {args.input_file} does not exist")
        return 1

    if not args.input_file.is_file():
        print(f"Error: {args.input_file} is not a file")
        return 1

    try:
        extract_messages_from_jsonl(args.input_file, args.output_file)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
