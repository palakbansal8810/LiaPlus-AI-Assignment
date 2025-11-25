# Sentiment Analysis Chatbot (Tier 1 + Full Tier 2 + Bonus)

## Features
- Real-time per-message sentiment analysis (positive/negative/neutral + score & confidence)
- Full conversation sentiment summary with trend detection (improving/declining/stable)
- Rich emotional journey summary using Llama-3.3-70B via Groq
- Inline sentiment badges under every user message
- Downloadable conversation log with complete analysis
- Clean, modern Streamlit interface

## Technologies Used
- Python + Streamlit
- Groq + Llama-3.3-70B (fast inference)
- Pydantic models for structured sentiment output
- Session state for persistent history

## How to Run
1. `pip install -r requirements.txt`
2. Get a free Groq API key at https://console.groq.com
3. Create `.env` file with `GROQ_API_KEY=your_key`
4. Run: `streamlit run streamlit_app.py`

## Sentiment Logic
- Each user message → analyzed with low-temperature LLM call → returns structured JSON
- After every exchange → full conversation re-analyzed:
  - Baseline heuristics (average score, trend)
  - Final refinement pass by LLM for natural-language summary and theme extraction

## Tier Completion
- Tier 1: Complete
- Tier 2: Complete + mood trend summary
- Bonus: Real-time display, export, polished UI
