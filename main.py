import os
from typing import List, Dict, Tuple
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# Pydantic Models for Structured Output
class SentimentAnalysis(BaseModel):
    """Sentiment analysis result for a single message"""
    sentiment: str = Field(description="Sentiment category: positive, negative, or neutral")
    score: float = Field(description="Sentiment score between -1 (very negative) and 1 (very positive)")
    confidence: float = Field(description="Confidence level between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the sentiment analysis")


class ConversationSummary(BaseModel):
    """Overall conversation sentiment summary"""
    overall_sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    average_score: float = Field(description="Average sentiment score across all messages")
    sentiment_trend: str = Field(description="Trend description: improving, declining, or stable")
    key_themes: List[str] = Field(description="Main emotional themes in the conversation")
    summary: str = Field(description="Brief summary of the conversation's emotional journey")


# Initialize Groq LLM
def init_llm(api_key: str = None) -> ChatGroq:
    """Initialize the Groq LLM"""
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7,
        max_tokens=500
    )


def init_sentiment_llm(api_key: str = None) -> ChatGroq:
    """Initialize a separate Groq LLM for sentiment analysis"""
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
        # Parse the response text as JSON to handle string numbers
        import json
        response_text = result.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Convert string numbers to float if needed
        score = float(data.get('score', 0))
        confidence = float(data.get('confidence', 0))
        
        return SentimentAnalysis(
            sentiment=data.get('sentiment', 'neutral'),
            score=score,
            confidence=confidence,
            reasoning=data.get('reasoning', 'Analysis completed')
        )
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
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
    
    # Build context from conversation history
    context = []
    for msg in conversation_history[-5:]:  # Last 5 exchanges for context
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

    user_prompt = f"""Previous conversation:
{context_str}

User's current message: "{user_message}"

Respond naturally and helpfully."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, I'm having trouble responding right now. Please try again."


def analyze_conversation_sentiment(
    conversation_history: List[Dict],
    sentiment_llm: ChatGroq
) -> ConversationSummary:
    """
    Tier 1: Analyze overall conversation sentiment
    
    Args:
        conversation_history: Complete conversation history
        sentiment_llm: LLM for sentiment analysis
        
    Returns:
        ConversationSummary with overall analysis
    """
    # Prepare conversation text
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
    
    # Calculate average sentiment score
    scores = [msg.get('sentiment', {}).get('score', 0) for msg in user_messages]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Determine trend
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
    
    # Determine overall sentiment based on average score
    if avg_score > 0.15:
        overall_sent = "positive"
    elif avg_score < -0.15:
        overall_sent = "negative"
    else:
        overall_sent = "neutral"
    
    # Extract themes from sentiment reasoning (rule-based approach)
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
    
    # Analyze all messages for themes
    for msg in user_messages:
        reasoning = msg.get('sentiment', {}).get('reasoning', '').lower()
        sentiment = msg.get('sentiment', {}).get('sentiment', '').lower()
        
        for keyword, theme in theme_keywords.items():
            if keyword in reasoning and theme not in themes:
                themes.append(theme)
                if len(themes) >= 4:
                    break
    
    # If no themes found from reasoning, use sentiment categories
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
    
    # Ensure we have at least one theme
    if not themes:
        themes = ['General conversation']
    
    # Create summary based on trend and scores
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
    
    return ConversationSummary(
        overall_sentiment=overall_sent,
        average_score=avg_score,
        sentiment_trend=trend,
        key_themes=themes[:4],  # Limit to 4 themes
        summary=summary
    )


def display_message(role: str, message: str, sentiment: SentimentAnalysis = None):
    """Display a formatted message with optional sentiment"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        print(f"\n[{timestamp}] USER: {message}")
        if sentiment:
            print(f"   └─ Sentiment: {sentiment.sentiment.upper()} "
                  f"(score: {sentiment.score:.2f}, confidence: {sentiment.confidence:.2f})")
            print(f"   └─ Reasoning: {sentiment.reasoning}")
    else:
        print(f"\n[{timestamp}] BOT: {message}")


def display_conversation_summary(summary: ConversationSummary):
    """Display the final conversation sentiment analysis"""
    print("\n" + "="*80)
    print("CONVERSATION SENTIMENT ANALYSIS")
    print("="*80)
    
    print(f"\nOverall Sentiment: {summary.overall_sentiment.upper()}")
    print(f"Average Score: {summary.average_score:.2f}")
    print(f"Sentiment Trend: {summary.sentiment_trend.upper()}")
    
    print(f"\nKey Themes:")
    for theme in summary.key_themes:
        print(f"   • {theme}")
    
    print(f"\nSummary:")
    print(f"   {summary.summary}")
    
    print("\n" + "="*80)


def save_conversation_log(
    conversation_history: List[Dict],
    summary: ConversationSummary,
    filename: str = None
):
    """Save conversation and analysis to a file"""
    if filename is None:
        filename = f"conversation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("CONVERSATION LOG\n")
        f.write("="*80 + "\n\n")
        
        for msg in conversation_history:
            f.write(f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['message']}\n")
            if msg['role'] == 'user' and 'sentiment' in msg:
                sent = msg['sentiment']
                f.write(f"   Sentiment: {sent['sentiment']} (score: {sent['score']:.2f})\n")
                f.write(f"   Reasoning: {sent['reasoning']}\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Sentiment: {summary.overall_sentiment}\n")
        f.write(f"Average Score: {summary.average_score:.2f}\n")
        f.write(f"Sentiment Trend: {summary.sentiment_trend}\n")
        f.write(f"\nKey Themes:\n")
        for theme in summary.key_themes:
            f.write(f"  • {theme}\n")
        f.write(f"\nSummary: {summary.summary}\n")
    
    print(f"\nConversation saved to: {filename}")


def run_chatbot():
    """Main function to run the chatbot"""
    print("="*80)
    print("SENTIMENT ANALYSIS CHATBOT")
    print("="*80)
    print("\nWelcome! I'm here to help you with any questions or concerns.")
    print("I'll analyze the sentiment of our conversation in real-time.")
    print("\nCommands:")
    print("  • Type 'quit' or 'exit' to end the conversation")
    print("  • Type 'help' for assistance")
    print("\n" + "-"*80)
    
    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nWarning: GROQ_API_KEY not found in environment variables")
        api_key = input("Please enter your Groq API key: ").strip()
    
    # Initialize LLMs
    try:
        llm = init_llm(api_key)
        sentiment_llm = init_sentiment_llm(api_key)
    except Exception as e:
        print(f"\nError initializing LLMs: {e}")
        print("Please check your API key and try again.")
        return
    
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nBot: Thank you for chatting with me. Analyzing our conversation...")
            break
        
        # Check for help
        if user_input.lower() == 'help':
            print("\nBot: I'm here to assist you! Feel free to:")
            print("   • Ask questions")
            print("   • Share concerns or feedback")
            print("   • Discuss any topic")
            print("   I'll respond helpfully while analyzing the sentiment of our conversation.")
            continue
        
        # Tier 2: Analyze sentiment of user message
        print("\nAnalyzing sentiment...", end="", flush=True)
        sentiment = analyze_sentiment(user_input, sentiment_llm)
        print("\r" + " "*30 + "\r", end="")  # Clear the line
        
        # Display user message with sentiment
        display_message("user", user_input, sentiment)
        
        # Store user message with sentiment
        conversation_history.append({
            'role': 'user',
            'message': user_input,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sentiment': {
                'sentiment': sentiment.sentiment,
                'score': sentiment.score,
                'confidence': sentiment.confidence,
                'reasoning': sentiment.reasoning
            }
        })
        
        # Generate bot response
        print("\nGenerating response...", end="", flush=True)
        bot_response = generate_chatbot_response(user_input, conversation_history, llm)
        print("\r" + " "*30 + "\r", end="")  # Clear the line
        
        # Display bot response
        display_message("bot", bot_response)
        
        # Store bot response
        conversation_history.append({
            'role': 'bot',
            'message': bot_response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Tier 1: Analyze overall conversation sentiment
    if len([m for m in conversation_history if m['role'] == 'user']) > 0:
        print("\nGenerating conversation analysis...")
        summary = analyze_conversation_sentiment(conversation_history, sentiment_llm)
        display_conversation_summary(summary)
        
        # Ask if user wants to save the log
        save = input("\nWould you like to save this conversation? (y/n): ").strip().lower()
        if save == 'y':
            save_conversation_log(conversation_history, summary)
    else:
        print("\nGoodbye! No messages to analyze.")
    
    print("\n" + "="*80)
    print("Thank you for using the Sentiment Analysis Chatbot!")
    print("="*80 + "\n")


if __name__ == "__main__":

    run_chatbot()
