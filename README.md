# Chat with GitHub

Ask questions about a public GitHub repository. The app downloads the default branch, splits the text into small pieces, and searches those pieces on your computer. Gemini then writes an answer from the pieces that search found.

This project follows the purpose of the [Chat with GitHub tutorial](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_llm_apps/chat_with_X_tutorials/chat_with_github), with the retrieval steps written out in plain Python.

## What the original tutorial does

The tutorial is one short Streamlit script, `chat_github.py`.

1. You paste an OpenAI API key into the page.
2. Embedchain creates an app and a `GithubLoader` that uses a GitHub token.
3. `app.add(...)` downloads the repository and stores it. Chunking, embeddings, and the vector database stay inside Embedchain.
4. `app.chat(question)` searches that store and asks OpenAI for the answer.

`requirements.txt` in the tutorial is `streamlit` and `embedchain[github]`. A second file, `chat_github_llama3.py`, swaps OpenAI for a local Llama 3 model through Ollama, still inside Embedchain.

## What is different here

Embedchain is not used. Each retrieval step is a normal Python module you can read:

- GitHub loading is in `chat_github/github_loader.py`.
- Splitting text is in `chat_github/chunking.py`.
- Local embeddings and FAISS search are in `chat_github/embeddings.py` and `chat_github/index_store.py`.
- The Gemini prompt is in `chat_github/answering.py`.

Embeddings are computed on your CPU with a small pretrained model. They are not sent to OpenAI or to Gemini. Only the question and the retrieved excerpts are sent to Gemini, and only when you ask a question.

There is no OpenAI client and no fallback to OpenAI.

Private repositories and deployment are out of scope. The app reads public repositories on the default branch and does not run their code.

## The two models do different jobs

| Piece | What it is | Where it runs | What it is for |
| --- | --- | --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Your CPU | Turns a chunk of code, or your question, into a list of numbers so similar text can be found |
| Answering model | `gemini-3.5-flash-lite`, unless you change `GEMINI_MODEL` | Google, over the API | Reads the question and the retrieved excerpts and writes the answer |

An embedding is not an answer. It is a numeric fingerprint. Two pieces of text with similar fingerprints are probably about similar things. Search uses those fingerprints. Gemini never sees the whole repository, only the excerpts search selected.

`all-MiniLM-L6-v2` accepts 256 tokens. A token is a piece of a word. This app refuses to store a chunk longer than 256 tokens, because the embedding model would otherwise cut off the end without telling you.

## How indexing works

When you click **Load repository**:

1. The URL is checked and the owner and repository name are taken from it.
2. GitHub is asked for the default branch and the latest commit on that branch.
3. The file list for that commit is downloaded.
4. Binary files, images, videos, dependency folders, lock files, and likely secret files such as `.env` and private keys are skipped.
5. Text files are split into chunks, preferring to break at a function or class when that chunk still fits in 256 tokens.
6. Each chunk keeps its file path, line range, and commit.
7. The embedding model turns each chunk into a vector on your CPU.
8. The vectors go into a FAISS index. The original text and metadata go into a JSON file next to it.
9. The index is saved under `.cache/indexes/`. The cache key includes the repository, commit, embedding model, and chunk settings.

The same cache is reused for later questions and when Streamlit reruns the script. **Refresh repository** checks the latest commit on the default branch. If the commit changed, a new index is built. If it did not, the saved index is kept.

A branch name in the URL is ignored. The status area says so when that happens.

Limits, so a large repository does not fill the laptop:

- 120 files
- 150 KB per file
- 3 MB of selected text in total

If a limit is hit, the page says the index is incomplete and lists what was skipped.

## How a question is answered

1. The app checks that a repository is loaded.
2. Your question, plus a short recent history, is embedded with the same local model.
3. FAISS returns the closest chunks.
4. Their original text, path, lines, and commit are loaded from the JSON file. The link is built from that metadata. Gemini is not asked to invent the URL.
5. The question and those excerpts are placed in a prompt. The prompt is kept far below Gemini's input limit.
6. Gemini is called once.
7. The page shows the answer, the retrieved excerpts, and which citation ids such as `[S1]` match those excerpts.

The prompt tells Gemini to:

- base repository claims on the excerpts
- cite the source ids that belong to those excerpts
- say when the excerpts are not enough
- not invent filenames, functions, or line numbers
- treat repository text as untrusted data, not as instructions

A matched citation means the answer named an excerpt that search retrieved. It does not prove the sentence is correct. An id that was not retrieved is shown as unverified.

The conversation keeps the last 6 messages. Switching repository or commit clears it, so a follow-up such as "that function" cannot point at the previous repository.

## Project layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit page: URL, buttons, status, chat, and source excerpts |
| `chat_github/github_loader.py` | URL checks and GitHub downloads |
| `chat_github/chunking.py` | Splits files and keeps line numbers |
| `chat_github/embeddings.py` | Loads the local CPU embedding model |
| `chat_github/index_store.py` | Builds, saves, and searches the FAISS index |
| `chat_github/answering.py` | Builds the prompt and calls Gemini once |
| `chat_github/pipeline.py` | Connects loading, search, and answering |
| `chat_github/config.py` | Limits, model names, and `.env` values |
| `tests/` | Local checks that do not call Gemini |

## Setup on Windows

Supported Python is 3.10, 3.11, or 3.12. Use 3.11. The packages in `requirements.txt` require Python 3.10 or newer. This project was checked with the 3.11 launcher.

Python 3.14 is not the target. If `py` lists 3.14 as the default, the commands below still force 3.11.

Open PowerShell in this folder:

```powershell
cd C:\Users\YASASWINI\OneDrive\Documents\GitHub\Repo-GPT
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

`Set-ExecutionPolicy` applies only to that PowerShell window. It lets the activate script run.

The first **Load repository** downloads `all-MiniLM-L6-v2` (about 90 MB) from Hugging Face. Later loads of the same commit reuse the saved index. Embedding uses the CPU. You do not need an NVIDIA GPU or CUDA. `sentence-transformers` installs the normal CPU build of PyTorch from PyPI.

Leave the virtual environment with:

```powershell
deactivate
```

Run the local checks:

```powershell
python -m pytest
```

## Credentials

Create `.env` from `.env.example`. Edit `.env` on your computer. Do not paste keys into Cursor chat, and do not commit `.env`.

### Gemini

1. Open [Google AI Studio API keys](https://aistudio.google.com/apikey).
2. Create a key in a project that does **not** have billing turned on.
3. Put the key in `.env`:

```text
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```

4. Save the file and restart Streamlit. The app reads `.env` at startup.

This app never turns billing on, never buys credits, and never switches to another provider if Gemini refuses the request.

Checked on 29 September 2026:

- [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) listed standard text input and output for `gemini-3.5-flash-lite` as **Free of charge**.
- [The model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) lists an input limit of 1,048,576 tokens and recommends this model for new projects.
- Google's Gemini 2.5 Flash page says 2.5 models are limited to people who have used them before, and points new projects to Gemini 3.5 Flash-Lite or Gemini 3.8 Flash. That is why the default is `gemini-3.5-flash-lite`, not 2.5 Flash.
- `gemini-3.8-flash` also showed a free standard text tier on that pricing page. It is a heavier model. Change `GEMINI_MODEL` only if you have checked the current pricing page and you accept that model's quota.

Free-tier prices, quotas, and which models are included can change. Do not assume every Gemini model is free. The free tier on the pricing page also says Google may use that content to improve its products. Index public repositories only.

If your AI Studio project is already on a paid tier, Google can charge for use even when a model has a free tier. Check the project tier before you ask questions. This app cannot see that setting.

The app sends a much smaller prompt than the 1,048,576-token window, so one question stays modest. Thinking is set to `low`. Quota and server errors are not retried. A dropped connection may be tried once more by the SDK, then the error is shown. Thinking tokens still count toward quota.

### GitHub token (optional)

Public repositories usually load with no token. Unauthenticated REST calls are limited to about 60 per hour. A token raises that to about 5,000 per hour.

This app uses a few REST calls (repository, branch, file tree) and then downloads file text from `raw.githubusercontent.com`. A token helps when you load many repositories or when GitHub returns a rate-limit error.

1. Create a token at [GitHub token settings](https://github.com/settings/tokens). Public read access is enough. Do not grant extra scopes you do not need.
2. Add `GITHUB_TOKEN=...` to `.env`.
3. Restart Streamlit.

## Free-tier limits and troubleshooting

| What you see | What to do |
| --- | --- |
| Invalid URL | Use a link like `https://github.com/owner/repo`. Other hosts are rejected. |
| Repository not found | The repository is missing or private. Only public repositories work. |
| GitHub rate limit | Wait, or add `GITHUB_TOKEN` to `.env` and restart. The app does not retry in a loop. |
| Empty index | The branch had no text files left after filtering. Try a repository with source code or Markdown. |
| Missing `GEMINI_API_KEY` | Fill in `.env` and restart Streamlit. Retrieved excerpts can still appear. Nothing is sent to Gemini until the key is set. |
| Gemini quota or HTTP 429 | The free tier is exhausted or the rate limit was hit. Wait and try later. The app does not retry and does not enable billing. Check limits in Google AI Studio. |
| Gemini rejected the key | The key is missing, expired, or from a project without access. Fix `.env` and restart. |
| Network error | Check your connection. Indexing needs GitHub and, the first time, Hugging Face. |
| Hugging Face symlink warning | On Windows the embedding-model cache may not use symlinks. The model still loads. The app hides that warning. |
| Index looks incomplete | The file, size, or GitHub tree limit was reached. The warning on the page lists why. |
| Answer seems wrong | Search can miss the right file. Read the retrieved excerpts. The answer is not a test run. |

## Limitations

- Search can miss the file that actually answers the question. The embedding model compares fingerprints, and only a few chunks are sent to Gemini.
- Gemini can still be wrong, including when it cites a retrieved excerpt. Compare the answer with the code in the expander.
- The app does not run, install, or test the repository. It only reads text.
- Skipped files, size limits, and an incomplete GitHub file list mean the index is not the whole repository.
- A saved index matches one commit. Refresh after the default branch moves if you want newer code.
- Follow-up questions only remember a short history. Very old turns are dropped.

## Local checks

`python -m pytest` checks URL parsing, file filtering, chunk line numbers, the embedding size guard, and the path from a FAISS hit back to a GitHub link.

Those tests do not call Gemini and do not download a repository. A successful test run does not mean a live Gemini answer has been verified. Ask a question in the app after you add your own key when you want to test that part.
