#!/usr/bin/env python3
"""
Synthetic data generation for linear probes on socioeconomic status, religion, and location.
Based on the TalkTuner-style next-token probing methodology.

Usage:
    1. Set your Anthropic API key as an environment variable:
       export ANTHROPIC_API_KEY="your-api-key-here"

    2. Run the script from the project root directory:
        cd user-models
        python src/generate_data.py

    3. Generated conversations will be saved as .txt files in the data/ directory:
       - socioeco_low_001.txt, socioeco_middle_***.txt, socioeco_high_***.txt
       - religion_christianity_***.txt, religion_hinduism_***.txt, religion_islam_***.txt
       - location_europe_***.txt, location_north_america_***.txt, location_east_asia_***.txt

    Each file contains a short conversation (3–5 exchanges). After the conversation,
    the final assistant line ends with an incomplete sentence to elicit the next-token prediction,
    e.g. for religion: "### Assistant: I think the user's religion is".
    Probes should read the hidden state at the last input token just before the next token is generated.
"""

import os
import argparse
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

RELIGIONS = ["christianity", "hinduism", "islam"]
LOCATIONS = ["europe", "north_america", "east_asia"]

# System prompt based on TalkTuner methodology
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

# Prompt templates for socioeconomic status
SOCIOECO_PROMPTS = [
    "Generate a conversation between a human user and an AI assistant. The socioeconomic status of this human user is {socioeco}. Make sure the conversation reflects this user's socioeconomic status. You may or may not include this user's socioeconomic status directly in the conversation. The user's response should start with '### Human:', and the AI assistant's response should start with '### Assistant:'. Output ONLY conversation lines prefixed by these tags — no other text, titles, or commentary. After the conversation, append one more assistant line that starts with exactly: '### Assistant: I think the user's socioeconomic status is' and end there (do not complete the sentence).",

    "Generate a conversation between a human user and an AI assistant. The socioeconomic status of this human user is {socioeco}. Make sure the conversation implicitly or explicitly reflects this user belongs to {class_name} class but not {other_class_name}. You may or may not include the user's socioeconomic status explicitly in the conversation. Be creative on the topic of the conversation. The user's response should start with '### Human:', and the AI assistant's response should start with '### Assistant:'. Output ONLY conversation lines prefixed by these tags — no other text, titles, or commentary. After the conversation, append one more assistant line that starts with exactly: '### Assistant: I think the user's socioeconomic status is' and end there (do not complete the sentence)."
]

# Prompt templates for religion
RELIGION_PROMPTS = [
    "Generate a conversation between a human user and an AI assistant. The user's religion is {religion}. Ensure the conversation reflects this attribute implicitly or explicitly through beliefs, practices, holidays, or cultural references, without being offensive or stereotypical. The conversation should be realistic and respectful. Make sure that the neither the assistant nor the user ever explicitly say the name of the religion. The user's messages start with '### Human:' and the assistant's with '### Assistant:'. Output ONLY conversation lines prefixed by these tags — no other text, titles, or commentary. After the conversation, append one more assistant line that starts with exactly: '### Assistant: I think the user's religion is' and end there (do not complete the sentence).",
    "Create a dialogue between a user and an assistant where the user's background includes {religion}. Let this influence the user's questions, experiences, and examples. Keep it natural and helpful. Use the same '### Human:' and '### Assistant:' prefixes. Output ONLY conversation lines prefixed by these tags — no other text, titles, or commentary. Make sure that neither the assistant nor the user explicitly says the name of the religion. After the conversation, append one more assistant line that starts with exactly: '### Assistant: I think the user's religion is' and end there (do not complete the sentence)."
]

# Prompt templates for location
LOCATION_PROMPTS = [
    "Generate a conversation between a human user and an AI assistant. The user is located (and presumably lives) in {location_human}. Make sure neither the user nor the assistant actually name the region {location_human}, though both can reference places in the region such as cities or states. Ensure the conversation reflects this through regional context like time zones, local customs, or place names, without being stereotypical. Use '### Human:' and '### Assistant:' prefixes. Output ONLY conversation lines prefixed by these tags — no other text, titles, or commentary. After the conversation, append one more assistant line that starts with exactly: '### Assistant: I think the user's location is' and end there (do not complete the sentence).",
    "Create a helpful, respectful conversation influenced by the user's location: {location_human}. Make sure neither the user nor the assistant actually name the region {location_human}, though both can reference places in the region such as cities or states. Maintain the '### Human:' and '### Assistant:' prefixes. Output ONLY conversation lines prefixed by these tags — no other text, titles, or commentary. After the conversation, append one more assistant line that starts with exactly: '### Assistant: I think the user's location is' and end there (do not complete the sentence)."
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

def generate_socioeco_data(num_per_status: int = 100) -> None:
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
                conversation = sanitize_conversation(conversation)
                conversation = ensure_tail(conversation, SOCIOECO_TAIL)
                save_conversation(conversation, "socioeco", socioeco, i)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Failed to generate conversation {i} for {socioeco}: {e}")
                continue

def generate_religion_data(num_per_value: int = 100) -> None:
    """Generate synthetic conversations for religion probing."""
    print("Generating religion data...")

    for religion in RELIGIONS:
        print(f"Generating {num_per_value} conversations for {religion}...")

        for i in range(num_per_value):
            template = random.choice(RELIGION_PROMPTS)
            prompt = template.format(religion=religion)

            try:
                conversation = generate_conversation(prompt)
                conversation = sanitize_conversation(conversation)
                conversation = ensure_tail(conversation, RELIGION_TAIL)
                save_conversation(conversation, "religion", religion, i)
                time.sleep(1)
            except Exception as e:
                print(f"Failed to generate conversation {i} for {religion}: {e}")
                continue


def generate_location_data(num_per_value: int = 100) -> None:
    """Generate synthetic conversations for location probing."""
    print("Generating location data...")

    location_names = {
        "europe": "Europe",
        "north_america": "North America",
        "east_asia": "East Asia",
    }

    for location in LOCATIONS:
        human_name = location_names[location]
        print(f"Generating {num_per_value} conversations for {human_name}...")

        for i in range(num_per_value):
            template = random.choice(LOCATION_PROMPTS)
            prompt = template.format(location_human=human_name)

            try:
                conversation = generate_conversation(prompt)
                conversation = sanitize_conversation(conversation)
                conversation = ensure_tail(conversation, LOCATION_TAIL)
                save_conversation(conversation, "location", location, i)
                time.sleep(1)
            except Exception as e:
                print(f"Failed to generate conversation {i} for {location}: {e}")
                continue

def main():
    """Main function to generate synthetic data. Use --task to select one."""
    parser = argparse.ArgumentParser(description="Generate synthetic conversations for probing")
    parser.add_argument(
        "--task",
        choices=["socioeco", "religion", "location", "all"],
        default="all",
        help="Which dataset to generate (default: all)",
    )
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    print("Starting synthetic data generation...")
    print(f"API Key present: {'ANTHROPIC_API_KEY' in os.environ}")

    per_socio = 100
    per_relig = 100
    per_loc = 100

    total = 0
    if args.task in ("socioeco", "all"):
        generate_socioeco_data(num_per_status=per_socio)
        total += len(SOCIOECONOMIC_STATUSES) * per_socio

    if args.task in ("religion", "all"):
        generate_religion_data(num_per_value=per_relig)
        total += len(RELIGIONS) * per_relig

    if args.task in ("location", "all"):
        generate_location_data(num_per_value=per_loc)
        total += len(LOCATIONS) * per_loc

    print("\nData generation complete!")
    print(f"Total files generated (expected): {total}")
SOCIOECO_TAIL = "### Assistant: I think the user's socioeconomic status is"
RELIGION_TAIL = "### Assistant: I think the user's religion is"
LOCATION_TAIL = "### Assistant: I think the user's location is"

def ensure_tail(conversation: str, tail: str) -> str:
    """Ensure the conversation ends exactly at the specified tail (no completion).
    If the model completed the tail, truncate it. If missing, append it.
    """
    text = conversation.replace("\r\n", "\n").strip()
    idx = text.rfind(tail)
    if idx != -1:
        return text[: idx + len(tail)]
    # Otherwise append the canonical tail as the final assistant line
    if not text.endswith("\n"):
        text += "\n"
    return text + tail

def sanitize_conversation(conversation: str) -> str:
    """Keep only properly prefixed conversation lines and drop preambles like
    "Sure! I'll write ...". Returns a cleaned conversation string.
    """
    text = (conversation or "").replace("\r\n", "\n").strip()
    lines = [ln.strip() for ln in text.split("\n")]
    kept = [ln for ln in lines if ln.startswith("### Human:") or ln.startswith("### Assistant:")]
    return "\n".join(kept).strip()

if __name__ == "__main__":
    main()
