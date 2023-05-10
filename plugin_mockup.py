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


def main(query, url):
    res = requests.post(url, json={
        "queries": [{"query": query, "filter": {"source": "file"}, "top_k": 10}]
    })
    docs = [r["text"] for r in res.json()["results"][0]["results"]]
    choices = chat_complete(
        build_prompt(query, docs),
        user_id='test',
        model=model,
        max_tokens_output=500,
        temperature=0.0,
    )
    return choices[0]


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


def build_prompt(query, docs):
    docs = "\n\n".join([f"chunk {i}: ```{doc}```" for i, doc in enumerate(docs)])
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a middle school mathematics tutor who answers student questions "
                "by parsing chunks of available text from mathematics textbooks, and then "
                "answering the question based on the most relevant text you can find. "
                "Your output should be formatted in JSON, formatted as follows: "
                "```{\"answer\": \"<your answer to the query>\", "
                "\"chunk\": <the text from the chunk on which you based your answer>}```."
                "Your answer should formatted in Latex. "
            )
        },
        {
            "role": "user",
            "content": (
                f"student query: ```{query}``` \n"
                "---\n"
                f"Textbook text chunks: \n```{docs}```\n"
            )
        }
    ]
    print(prompt_messages[1]["content"])
    return prompt_messages


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Siyavula GPT query-answer mockup.")
    parser.add_argument("query", type=str, help="Query to submit to ChatGPT.")
    parser.add_argument("url", type=str, help="URL of the retrieval plugin API.")
    args = parser.parse_args()
    print(main(args.query, args.url))
    # print(json.dumps(json.loads(), indent=4))
