# YouTube Fact Checker

A command-line tool that automatically fact-checks claims made in YouTube videos by analyzing their transcripts, identifying factual claims, and verifying them against web search results.

## Features

- Extracts transcripts from YouTube videos
- Uses AI (OpenAI or Gemini) to identify factual claims in the transcript
- Verifies each claim using web search (via SerperAPI)
- Generates a comprehensive fact-checking report with:
  - Verdict for each claim (TRUE, FALSE, PARTIALLY TRUE, UNVERIFIABLE)
  - Explanation and supporting sources
  - Overall reliability score
  - Summary of the video's factual accuracy

## Requirements

- Python 3.12 or higher
- API keys for:
  - YouTube Data API
  - OpenAI API (for GPT-4o) or Google Gemini API
  - SerperAPI (for web search)

## Setup with uv

This project uses [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

### 1. Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the repository

```bash
git clone https://github.com/yourusername/fact-check.git
cd fact-check
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Set up environment variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:
- `YOUTUBE_API_KEY`: Get from [Google Cloud Console](https://console.cloud.google.com/)
- `OPENAI_API_KEY`: Get from [OpenAI Platform](https://platform.openai.com/)
- `GEMINI_API_KEY`: Get from [Google AI Studio](https://makersuite.google.com/)
- `SERPER_API_KEY`: Get from [SerperAPI](https://serper.dev/)

## Usage

Run the fact-checker with a YouTube URL:

```bash
uv run main.py https://www.youtube.com/watch?v=VIDEO_ID
```

### Options

- `--llm`: Choose the LLM provider (`openai` or `gemini`, default: `openai`)
- `--output`: Specify an output file path for the JSON results
- `--debug`: Enable debug mode with detailed error messages

Example:

```bash
uv run main.py https://www.youtube.com/watch?v=VIDEO_ID --llm gemini --output results.json
```

## How It Works

1. The tool extracts the video transcript using the YouTube Transcript API
2. It uses an LLM (OpenAI GPT-4o or Google Gemini) to identify factual claims in the transcript
3. For each claim, it performs a web search using SerperAPI to find relevant information
4. The LLM analyzes the search results to determine the veracity of each claim
5. Finally, it generates a summary of the fact-checking results, including an overall reliability score

## Output

The tool provides a detailed report in the console, including:
- Video metadata (title, channel, publish date, view count)
- Summary of fact-checking results
- Reliability score (0.00-1.00)
- List of claims with verdicts and explanations

If an output file is specified, a detailed JSON report is also saved.
