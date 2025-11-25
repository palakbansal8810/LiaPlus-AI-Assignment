import os
from typing import List, Dict, Optional
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

def inject_custom_css():
   
    st.markdown(
        """
        <style>
        div[data-testid="stChatMessage"] p {
            font-size: 0.92rem !important;
            line-height: 1.35 !important;
        }
        div[data-testid="stChatMessage"] {
            padding: 0.35rem 0.6rem !important;
        }
        div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
            width: 32px !important;
            height: 32px !important;
        }
        .small-sentiment {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.75rem;
            font-weight: 500;
            color: #f8fafc;
            background: #1f2937;
            border-radius: 999px;
            padding: 0.15rem 0.7rem;
            margin-top: 0.35rem;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        }
        .small-sentiment span {
            font-weight: 400;
            opacity: 0.85;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

load_dotenv()


class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="Sentiment category: positive, negative, or neutral")
    score: float = Field(description="Sentiment score between -1 (very negative) and 1 (very positive)")
    confidence: float = Field(description="Confidence level between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the sentiment analysis")


class ConversationSummary(BaseModel):
 
    overall_sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    average_score: float = Field(description="Average sentiment score across all messages")
    sentiment_trend: str = Field(description="Trend description: improving, declining, or stable")
    key_themes: List[str] = Field(description="Main emotional themes in the conversation")
    summary: str = Field(description="Brief summary of the conversation's emotional journey")

def init_llm(api_key: str = None) -> ChatGroq:
   
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7,
        max_tokens=500
    )

def init_sentiment_llm(api_key: str = None) -> ChatGroq:
   
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
        max_tokens=300
    )

def analyze_sentiment(message: str, sentiment_llm: ChatGroq) -> SentimentAnalysis:
   
    prompt = f"""Analyze the sentiment of the following message. Consider:
- Overall emotional tone (positive, negative, neutral)
- Intensity of emotion
- Context and nuance

Message: "{message}"

Provide a sentiment score where:
- Positive emotions: 0.1 to 1.0
- Neutral: -0.1 to 0.1
- Negative emotions: -1.0 to -0.1

IMPORTANT: Return ONLY valid JSON with numeric values for score and confidence (not strings).

Return in this exact format:
{{
  "sentiment": "positive" or "negative" or "neutral",
  "score": 0.5,
  "confidence": 0.8,
  "reasoning": "brief explanation"
}}"""
    try:
        result = sentiment_llm.invoke([HumanMessage(content=prompt)])
        import json
        response_text = result.content.strip()
        
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        data = json.loads(response_text)
        
        score = float(data.get('score', 0))
        confidence = float(data.get('confidence', 0))
        
        return SentimentAnalysis(
            sentiment=data.get('sentiment', 'neutral'),
            score=score,
            confidence=confidence,
            reasoning=data.get('reasoning', 'Analysis completed')
        )
    except Exception:
        return SentimentAnalysis(
            sentiment="neutral",
            score=0.0,
            confidence=0.0,
            reasoning="Error in analysis"
        )


def generate_chatbot_response(
    user_message: str,
    conversation_history: List[Dict],
    llm: ChatGroq
) -> str:
    """
    Generate chatbot response based on conversation history
    
    Args:
        user_message: Current user message
        conversation_history: List of previous messages
        llm: Language model for response generation
        
    Returns:
        Chatbot's response text
    """
    context = []
    for msg in conversation_history[-5:]:
        if msg['role'] == 'user':
            context.append(f"User: {msg['message']}")
        else:
            context.append(f"Assistant: {msg['message']}")
    
    context_str = "\n".join(context) if context else "No previous context"
    
    system_prompt = """You are a helpful, empathetic customer service chatbot. 
Your role is to:
- Listen carefully to user concerns
- Respond with empathy and understanding
- Provide helpful and constructive responses
- Acknowledge emotions when appropriate
- Keep responses concise (2-3 sentences)

Be natural, friendly, and professional."""
    if context_str!="No previous context":
        user_prompt = f"""Previous conversation:
    {context_str}

    User's current message: "{user_message}"

    Respond naturally and helpfully."""
    else:   
        user_prompt = f"""Greet the user with a message like "Hi, how can I help you today?"
        User's current message: "{user_message}"
        Respond naturally and helpfully."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception:
        return "I apologize, I'm having trouble responding right now. Please try again."


def analyze_conversation_sentiment(
    conversation_history: List[Dict],
    sentiment_llm: ChatGroq
) -> ConversationSummary:
 
    user_messages = [
        msg for msg in conversation_history 
        if msg['role'] == 'user'
    ]
    
    if not user_messages:
        return ConversationSummary(
            overall_sentiment="neutral",
            average_score=0.0,
            sentiment_trend="stable",
            key_themes=[],
            summary="No user messages to analyze"
        )
    scores = [msg.get('sentiment', {}).get('score', 0) for msg in user_messages]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    if len(scores) >= 2:
        first_half_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
        second_half_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        
        if second_half_avg - first_half_avg > 0.2:
            trend = "improving"
        elif first_half_avg - second_half_avg > 0.2:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"
    
    if avg_score > 0.15:
        overall_sent = "positive"
    elif avg_score < -0.15:
        overall_sent = "negative"
    else:
        overall_sent = "neutral"
    
    themes = []
    theme_keywords = {
        'frustration': 'Frustration and difficulty',
        'confusion': 'Confusion and uncertainty',
        'difficulty': 'Problem-solving challenges',
        'negative': 'Negative emotional tone',
        'disappointment': 'Disappointment',
        'gratitude': 'Gratitude and appreciation',
        'satisfaction': 'Satisfaction and resolution',
        'positive': 'Positive emotional tone',
        'happiness': 'Happiness and contentment',
        'thanks': 'Appreciation',
        'help': 'Seeking assistance'
    }
    
    for msg in user_messages:
        reasoning = msg.get('sentiment', {}).get('reasoning', '').lower()
        sentiment = msg.get('sentiment', {}).get('sentiment', '').lower()
        
        for keyword, theme in theme_keywords.items():
            if keyword in reasoning and theme not in themes:
                themes.append(theme)
                if len(themes) >= 4:
                    break
    
    if not themes:
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for msg in user_messages:
            sent = msg.get('sentiment', {}).get('sentiment', 'neutral')
            sentiment_counts[sent] += 1
        
        if sentiment_counts['negative'] > 0:
            themes.append('Challenges and concerns')
        if sentiment_counts['positive'] > 0:
            themes.append('Positive resolution')
        if sentiment_counts['neutral'] > 0:
            themes.append('Information seeking')
    
    if not themes:
        themes = ['General conversation']
    
    num_messages = len(user_messages)
    
    if trend == "improving":
        if scores[0] < -0.3 and scores[-1] > 0.3:
            summary = f"The conversation showed significant improvement from initial negative sentiment (score: {scores[0]:.2f}) to a positive conclusion (score: {scores[-1]:.2f}). User concerns were successfully addressed over {num_messages} messages."
        else:
            summary = f"The conversation demonstrated an improving emotional trajectory across {num_messages} messages, starting at {scores[0]:.2f} and improving to {scores[-1]:.2f}."
    elif trend == "declining":
        summary = f"The conversation started more positively (score: {scores[0]:.2f}) but sentiment declined toward the end (score: {scores[-1]:.2f}) over {num_messages} messages."
    else:
        if avg_score > 0.3:
            summary = f"The conversation maintained a consistently positive tone throughout {num_messages} messages, with an average sentiment score of {avg_score:.2f}."
        elif avg_score < -0.3:
            summary = f"The conversation reflected ongoing challenges with a consistently negative tone across {num_messages} messages (average score: {avg_score:.2f})."
        else:
            summary = f"The conversation maintained a relatively neutral emotional tone across {num_messages} messages, with an average sentiment score of {avg_score:.2f}."
    
    baseline_summary = ConversationSummary(
        overall_sentiment=overall_sent,
        average_score=avg_score,
        sentiment_trend=trend,
        key_themes=themes[:4],
        summary=summary
    )
    
    transcript_lines = []
    for msg in conversation_history[-20:]:
        display_role = msg['role'].upper()
        line = f"{display_role}: {msg['message']}"
        if msg['role'] == 'user' and 'sentiment' in msg:
            sent = msg['sentiment']
            line += (
                f" (sentiment={sent.get('sentiment')}, "
                f"score={float(sent.get('score', 0)):.2f}, "
                f"confidence={float(sent.get('confidence', 0)):.2f})"
            )
        transcript_lines.append(line)
    transcript_text = "\n".join(transcript_lines) if transcript_lines else "No transcript available."
    
    theme_hint = ", ".join(baseline_summary.key_themes)
    system_prompt = (
        "You are an expert conversation analyst. "
        "Summarize emotional tone and key themes. "
        "Respond with STRICT JSON matching the schema."
    )
    user_prompt = f"""
Conversation transcript:
{transcript_text}

Baseline metrics:
- Average sentiment score: {baseline_summary.average_score:.3f}
- Trend hint: {baseline_summary.sentiment_trend}
- Theme hints: {theme_hint or "None"}
- Message count: {num_messages}

Return JSON with:
{{
  "overall_sentiment": "positive|negative|neutral",
  "average_score": number between -1 and 1,
  "sentiment_trend": "improving|declining|stable",
  "key_themes": ["theme1", "theme2"],
  "summary": "two concise sentences"
}}
"""
    try:
        response = sentiment_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt.strip())
        ])
        import json
        response_text = response.content.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()
        data = json.loads(response_text)
        key_themes = data.get('key_themes', [])
        if isinstance(key_themes, str):
            key_themes = [item.strip() for item in key_themes.split(',') if item.strip()]
        return ConversationSummary(
            overall_sentiment=data.get('overall_sentiment', baseline_summary.overall_sentiment),
            average_score=float(data.get('average_score', baseline_summary.average_score)),
            sentiment_trend=data.get('sentiment_trend', baseline_summary.sentiment_trend),
            key_themes=key_themes[:4] if key_themes else baseline_summary.key_themes,
            summary=data.get('summary', baseline_summary.summary)
        )
    except Exception:
        return baseline_summary


def format_conversation_export(conversation_history: List[Dict],summary: Optional[ConversationSummary]) -> str:
    lines = [
        "CONVERSATION LOG",
        "=" * 80,
        ""
    ]
    for msg in conversation_history:
        lines.append(f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['message']}")
        if msg['role'] == 'user' and 'sentiment' in msg:
            sent = msg['sentiment']
            lines.append(
                f"   Sentiment: {sent['sentiment']} (score: {sent['score']:.2f}, confidence: {sent['confidence']:.2f})"
            )
            lines.append(f"   Reasoning: {sent['reasoning']}")
        lines.append("")
    
    if summary:
        lines.extend([
            "=" * 80,
            "OVERALL ANALYSIS",
            "=" * 80,
            "",
            f"Overall Sentiment: {summary.overall_sentiment}",
            f"Average Score: {summary.average_score:.2f}",
            f"Sentiment Trend: {summary.sentiment_trend}",
            "",
            "Key Themes:"
        ])
        for theme in summary.key_themes:
            lines.append(f"  • {theme}")
        lines.extend([
            "",
            f"Summary: {summary.summary}"
        ])
    
    return "\n".join(lines)
def init_streamlit_state(): 
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "conversation_summary" not in st.session_state:
        st.session_state.conversation_summary = None
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = os.getenv("GROQ_API_KEY", "")


def ensure_llms(api_key: str):
    if not api_key:
        raise ValueError("A Groq API key is required.")
    
    needs_refresh = (
        "llm" not in st.session_state or
        "sentiment_llm" not in st.session_state or
        st.session_state.get("llm_api_key") != api_key
    )
    
    if needs_refresh:
        st.session_state.llm = init_llm(api_key)
        st.session_state.sentiment_llm = init_sentiment_llm(api_key)
        st.session_state.llm_api_key = api_key


def reset_conversation():
    st.session_state.conversation_history = []
    st.session_state.conversation_summary = None

def append_user_message(message: str, sentiment: SentimentAnalysis):
    st.session_state.conversation_history.append({
        "role": "user",
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment": {
            "sentiment": sentiment.sentiment,
            "score": sentiment.score,
            "confidence": sentiment.confidence,
            "reasoning": sentiment.reasoning
        }
    })
def append_bot_message(message: str):
    st.session_state.conversation_history.append({
        "role": "bot",
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def render_sidebar() -> str:
    with st.sidebar:
        st.header("Controls")
        st.caption("Provide your Groq API key to enable the chatbot.")
        
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            key="api_key_input"
        ).strip()
        
        if st.button("Clear conversation", use_container_width=True):
            reset_conversation()
            st.rerun()
        
        summary = st.session_state.conversation_summary
        history = st.session_state.conversation_history
        if history:
            export_text = format_conversation_export(history, summary)
            st.download_button(
                "Download chat log",
                data=export_text,
                file_name=f"conversation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
    return api_key

def render_conversation():
    for msg in st.session_state.conversation_history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["message"])
            if msg["role"] == "user" and "sentiment" in msg:
                sent = msg["sentiment"]
                sentiment_html = (
                    "<div class='small-sentiment'>"
                    f"<strong>{sent['sentiment'].title()}</strong>"
                    f"<span>score {sent['score']:.2f} · conf {sent['confidence']:.2f}</span>"
                    "</div>"
                )
                st.markdown(sentiment_html, unsafe_allow_html=True)

def render_summary_panel():

    summary: Optional[ConversationSummary] = st.session_state.conversation_summary
    
    if not summary:
        st.info("Start chatting to see real-time sentiment insights.")
        return
    
    if st.button("Show last summary & key themes", use_container_width=True):
        st.markdown(f"**Summary:** {summary.summary}")
        if summary.key_themes:
            st.markdown(f"**Key themes:** {', '.join(summary.key_themes)}")
        else:
            st.markdown("**Key themes:** No themes detected yet.")


def process_user_message(user_message: str, api_key: str):
    try:
        ensure_llms(api_key)
    except Exception as exc:
        st.error(f"Unable to initialize Groq models: {exc}")
        return
    
    with st.spinner("Analyzing sentiment..."):
        sentiment = analyze_sentiment(user_message, st.session_state.sentiment_llm)
    
    
    with st.spinner("Generating response..."):
        bot_reply = generate_chatbot_response(
            user_message,
            st.session_state.conversation_history,
            st.session_state.llm
        )
    
    append_user_message(user_message, sentiment)
    append_bot_message(bot_reply)
    st.session_state.conversation_summary = analyze_conversation_sentiment(
        st.session_state.conversation_history,
        st.session_state.sentiment_llm
    )
    st.rerun()

def main():
    """Streamlit entry point."""
    st.set_page_config(
        page_title="Sentiment Analysis Chatbot",
        layout="wide"
    )
    
    inject_custom_css()
    st.title("Sentiment Analysis Chatbot")
    st.caption("Chat with an empathetic assistant while monitoring real-time sentiment.")
    
    init_streamlit_state()
    api_key = render_sidebar()
    
    render_conversation()
    render_summary_panel()
    
    prompt = st.chat_input("How can I help you today?")
    if prompt:
        if not api_key:
            st.warning("Please provide your Groq API key in the sidebar first.")
        else:
            process_user_message(prompt, api_key)

if __name__ == "__main__":
    main()