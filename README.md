# ChatTo

Chat to resources LLMs don't have access to.

## What is ChatTo

ChatTo is a Python application that utilises [embeddings](https://platform.openai.com/docs/guides/embeddings) and [langchain](https://github.com/hwchase17/langchain) to ask questions on contents, including PDF files, wikipedia, source code.
- You can upload a PDF file, the play [The Tragedy of Hamlet](hamlet.pdf) has been preloaded for demo purposes, and learn about the content of PDF using Q&A.
- You can also provide a topic you'd like to know on [Wikipedia](https://en.wikipedia.org/wiki/Main_Page), this would actually perform a live search on wiki and prepare the content for answering your questions.
- You can ask the program to read source code and produce code snippets, a libray called [Pandas AI](pandas-ai) has been preloaded for demo.

## Why ChatTo

The biggest pain point this project is looking to solve when using LLMs is context. There is a lot of data either held by individuals or companies sitting there silently, because LLMs can't get access, or the data can't be made available for training LLMs. This project investigates turning these data into context for LLMs, so that you can leverage pre-trained LLM to perform tasks you couldn't before, whilst keeping your data private to you.

Other pain points with LLMs, specifically related to ChatGPT users without Plus subscription:
- Outdated data: the data used to train current version of ChatGPT is up until September 2021. This project can make additional information beyond the time window accessible to LLMs.
- No internet connection: aside from outdated data, LLMs do not access to the internet. This project can also scrap and provide information on the internet as additional context to LLMs.

## How to run the app

### Prerequisite

You have python3 installed or run the following command to install it.
```
brew install python@3.10
```

### Install dependencies

```
pip install -r requirements.txt
```

### Retrieve your OpenAI API key

```
https://platform.openai.com/account/api-keys
```

### Insert OpenAI API key to `.env`

```
echo "OPENAI_API_KEY='your-api-key'" >> ~/.env
```

### Start the application

```
streamlit run app.py
```

## One More Thing

It's great to be able to load in extra information to LLMs and Q&A on it, but how well this approach works, how could you even know?

To solve this problem, I have also included an `evaluation` script that assert answers from LLMs against answers from [hand picked resource](hamlet_qna.pdf).

To run the evaluation program:
```
python3 evaluation.py
```
The results are stored in [predications](predications.json) and [evaluation](evaluation.json).