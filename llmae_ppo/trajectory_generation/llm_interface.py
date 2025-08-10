"""
LLM Interface for trajectory generation.
Provides a clean interface to various LLM providers with error handling and retries.
"""

from typing import Optional

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMInterface:
    """Interface for LLM API calls with error handling and retries."""

    def __init__(
        self,
        model: str = "meta-llama/llama-3.3-70b-instruct",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize LLM interface.

        Args:
            model: Model name to use
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 200,
    ) -> str:
        """
        Query the LLM with retries and error handling.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text

        Raises:
            Exception: If all retries fail
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    print(
                        f"LLM query failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    print(f"LLM query failed after {self.max_retries} attempts")

        raise last_exception
