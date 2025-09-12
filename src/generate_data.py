#!/usr/bin/env python3
"""
Synthetic data generation for linear probes on socioeconomic status and happiness levels.
Based on the TalkTuner project methodology.

Usage:
    1. Set your Anthropic API key as an environment variable:
       export ANTHROPIC_API_KEY="your-api-key-here"

    2. Run the script from the project root directory:
       cd user-models
       python src/generate_data.py

    3. Generated conversations will be saved as .txt files in the data/ directory:
       - socioeco_low_001.txt, socioeco_low_002.txt, etc.
       - socioeco_middle_001.txt, socioeco_middle_002.txt, etc.
       - socioeco_high_001.txt, socioeco_high_002.txt, etc.
       - happiness_sad_001.txt, happiness_sad_002.txt, etc.
       - happiness_neutral_001.txt, happiness_neutral_002.txt, etc.
       - happiness_happy_001.txt, happiness_happy_002.txt, etc.

    Each file contains a complete conversation with 3-5 turns between human and assistant.
    By default, generates 50 conversations per category (300 total files).
"""

import os
import random
from anthropic import Anthropic
import time
from typing import List, Dict, Any

# Initialize Anthropic client
client = Anthropic()

# Define categories for synthetic data generation
SOCIOECONOMIC_STATUSES = {
    "low": {
        "class_name": "lower",
        "other_class_name": "middle or upper classes"
    },
    "middle": {
        "class_name": "middle",
        "other_class_name": "lower or upper classes"
    },
    "high": {
        "class_name": "upper",
        "other_class_name": "lower or middle classes"
    }
}

HAPPINESS_LEVELS = ["sad", "neutral", "happy"]

# System prompt based on TalkTuner methodology
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

# Prompt templates for socioeconomic status
SOCIOECO_PROMPTS = [
    "Generate a conversation between a human user and an AI assistant. The socioeconomic status of this human user is {socioeco}. Make sure the conversation reflects this user's socioeconomic status. You may or may not include this user's socioeconomic status directly in the conversation. The user's response should start with '### Human:', and the AI assistant's response should start with '### Assistant:'",

    "Generate a conversation between a human user and an AI assistant. The socioeconomic status of this human user is {socioeco}. Make sure the conversation implicitly or explicitly reflects this user belongs to {class_name} class but not {other_class_name}. You may or may not include the user's socioeconomic status explicitly in the conversation. Be creative on the topic of the conversation. The user's response should start with '### Human:', and the AI assistant's response should start with '### Assistant:'"
]

# Prompt templates for happiness levels
HAPPINESS_PROMPTS = [
    "Generate a conversation between a human user and an AI assistant. The emotional state of this human user is {happiness}. Make sure the conversation reflects this user's emotional state through their tone, word choice, and topics they discuss. You may or may not include this user's emotional state directly in the conversation. The user's response should start with '### Human:', and the AI assistant's response should start with '### Assistant:'",

    "Generate a conversation between a human user and an AI assistant. This human user is feeling {happiness}. Make sure the conversation reflects this emotional state through the way they communicate and the topics they bring up. Be creative on the topics of conversation. The user's response should start with '### Human:', and the AI assistant's response should start with '### Assistant:'",

    "Generate a conversation between a human user and an AI assistant. The human user's current mood is {happiness}. Make sure this emotional state comes through naturally in their conversation style and the subjects they choose to discuss. The conversation should be 3-5 exchanges long. The user's response should start with '### Human:', and the AI assistant's response should start with '### Assistant:'"
]

def generate_conversation(prompt: str, max_retries: int = 3) -> str:
    """Generate a single conversation using Claude Haiku."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e

def save_conversation(conversation: str, category: str, value: str, index: int) -> None:
    """Save a conversation to a text file."""
    filename = f"{category}_{value}_{index:03d}.txt"
    filepath = os.path.join("data", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(conversation)

    print(f"Saved: {filename}")

def generate_socioeco_data(num_per_status: int = 50) -> None:
    """Generate synthetic conversations for socioeconomic status probing."""
    print("Generating socioeconomic status data...")

    for socioeco, details in SOCIOECONOMIC_STATUSES.items():
        print(f"Generating {num_per_status} conversations for {socioeco} socioeconomic status...")

        for i in range(num_per_status):
            # Randomly select prompt template
            template = random.choice(SOCIOECO_PROMPTS)
            prompt = template.format(
                socioeco=socioeco,
                class_name=details["class_name"],
                other_class_name=details["other_class_name"]
            )

            try:
                conversation = generate_conversation(prompt)
                save_conversation(conversation, "socioeco", socioeco, i)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Failed to generate conversation {i} for {socioeco}: {e}")
                continue

def generate_happiness_data(num_per_level: int = 50) -> None:
    """Generate synthetic conversations for happiness level probing."""
    print("Generating happiness level data...")

    for happiness in HAPPINESS_LEVELS:
        print(f"Generating {num_per_level} conversations for {happiness} emotional state...")

        for i in range(num_per_level):
            # Randomly select prompt template
            template = random.choice(HAPPINESS_PROMPTS)
            prompt = template.format(happiness=happiness)

            try:
                conversation = generate_conversation(prompt)
                save_conversation(conversation, "happiness", happiness, i)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Failed to generate conversation {i} for {happiness}: {e}")
                continue

def main():
    """Main function to generate all synthetic data."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    print("Starting synthetic data generation...")
    print(f"API Key present: {'ANTHROPIC_API_KEY' in os.environ}")

    # Generate data for both categories
    generate_socioeco_data(num_per_status=50)
    generate_happiness_data(num_per_level=50)

    print("\nData generation complete!")
    print(f"Total files generated: {len(SOCIOECONOMIC_STATUSES) * 50 + len(HAPPINESS_LEVELS) * 50}")

if __name__ == "__main__":
    main()
