import tiktoken
import requests
import os
import argparse
import json
import time

from typing import List, Tuple, Dict, Any

model = {
    "name": "gpt-3.5-turbo",
    "token_limit": 4096,
    "url": "https://api.openai.com/v1/chat/completions"
}


def main(filepath, url):
    fnames = os.listdir(filepath)
    for fname in fnames:
        if (not fname.endswith('.html')) or (not "whole-numbers" in fname):
            continue
        print("Indexing:", fname)
        with open(os.path.join(filepath, fname), 'r') as f:
            text_html = f.read()

        num_tokens_input = num_tokens(model, text_html)
        half_limit = (model["token_limit"] - 100) // 2
        if num_tokens_input > half_limit:
            n_chunks = (num_tokens_input // half_limit) + 1
            chunks = chunk_tokens(text_html, model, half_limit, n_chunks)
        else:
            chunks = [text_html]

        docs = []
        for s in chunks:
            prompt = html_to_latex_prompt(s)
            choices = chat_complete(
                prompt,
                user_id='test',
                model=model, max_tokens_output=half_limit,
                temperature=0.0,
            )
            # print(choices[0])
            text = choices[0].strip()
            print(text)
            # print(s, "*" * 50, text, sep="\n")
            print("-----------------------------------")
            docs.append({"text": text, "metadata": {"source": "file"}})
            # sleep for 15 seconds to avoid rate limit
            time.sleep(15)
        res = requests.post(url, json={"documents": docs})
        print(res.status_code, res.json())


def chat_complete(
        messages: str,  # ~prompt
        user_id: str,
        model: Dict[str, str],
        max_tokens_output: int = 500,  # output
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: List[str] = None,
        n: int = 1,
) -> str:
    """
    Given a list of messages constituting a prompt, call the OpenAI
    ChatGPT Completion endpoint.

    args
    ----
    messages: list - list of messages constituting the prompt
    user_id: str - user id of the user making the request
    model: dict - model to use for completion
    max_tokens_output: int - maximum number of tokens to return in the output
    temperature: float - temperature of the model
    top_p: float - see OpenAI docs
    frequency_penalty: float - see OpenAI docs
    presence_penalty: float - see OpenAI docs
    stop: list - list of strings to stop completion on
    n: int - number of completions to return

    returns
    -------
    choices: list - list of completions
    """
    # Check input
    input_text = " ".join([message["content"] for message in messages])
    num_tokens_input = num_tokens(model, input_text)
    print(num_tokens_input, max_tokens_output)
    if num_tokens_input > model["token_limit"] - max_tokens_output:
        raise ValueError("Total number of tokens (input + output) exceeds model context limit.")

    # Perform completion
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    json_data = {
        "model": model["name"],
        "messages": messages,
        "max_tokens": max_tokens_output,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "user": str(user_id),
        "stop": stop,
        "n": n,
    }
    response = requests.post(
        model["url"],
        headers=headers,
        json=json_data,
    )
    response.raise_for_status()
    choices = [
        item["message"]["content"].strip() for item in response.json()["choices"]
    ]
    return choices


def num_tokens(
        model: Dict[str, str],
        string: str
) -> int:
    """
    Returns the number of tokens in a string for a given model.

    args
    ----
    model: dict - model dict
    string: str - string to be tokenised

    returns
    -------
    num_tokens: int - number of tokens in string
    """
    encoding = tiktoken.encoding_for_model(model["name"])
    tokens = encoding.encode(string)
    return len(tokens)


def chunk_tokens(
        text: str,
        model: Dict[str, str],
        n: int,
        num_chunks: int = 1,
) -> str:
    """
    Trims a string to a given number of tokens.

    args
    ----
    text: str - string to be trimmed
    model: dict - model dict
    n: int - number of tokens to trim to

    returns
    -------
    text: str - trimmed string
    """
    chunks = []
    encoding = tiktoken.encoding_for_model(model["name"])
    tokens = encoding.encode(text)
    for i in range(num_chunks):
        decoded_tokens = [
            encoding.decode_single_token_bytes(token)
            for token in tokens[n * i:(n * i) + n]
        ]
        print("chunk len", len(decoded_tokens))
        chunks.append(b''.join(decoded_tokens).decode("utf-8", errors="ignore"))
    return chunks


def html_to_latex_prompt(html):
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that extracts text from HTML "
                "(i.e. removing all HTML tags etc.), preserving all equations "
                "in Latex."
            )
        },
        {
            "role": "user",
            "content": (
                "Given the following HTML (contained in triple backticks), "
                "extract the text, preserving all equations in Latex. "
                f"Input HTML: ```{html}```"
            )
        }
    ]
    return prompt_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to upsert documents from folder to the vector DB via the retrieval plugin.")
    parser.add_argument("filepath", type=str, help="Path to folder containing HTML documents.")
    parser.add_argument("url", type=str, help="URL of the retrieval plugin API.")
    args = parser.parse_args()
    main(args.filepath, args.url)
