import os
import re 
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
from langgraph.graph import StateGraph
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import json

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_PATH = "data/reviews.csv"
KAGGLE_DATASET = "farukalam/yelp-restaurant-reviews"


# ---------- Auto-download Kaggle dataset ----------
def ensure_kaggle_dataset():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("📥 Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path="data/")
        
        # Find and unzip the file
        for file in os.listdir("data"):
            if file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join("data", file), 'r') as zip_ref:
                    zip_ref.extractall("data")
                os.remove(os.path.join("data", file))
        
        # Find the extracted CSV and rename to reviews.csv
        for file in os.listdir("data"):
            if file.endswith(".csv") and file != "reviews.csv":
                os.rename(os.path.join("data", file), DATA_PATH)
        print("✅ Dataset ready.")



# ---------- Groq API Helper ----------
def ask_groq(prompt: str, model="llama3-8b-8192"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()



# ---------- Agent 1: Feedback Response ----------
def feedback_response_agent(state):
    review_text = state["input"]
    sentiment_prompt = f"""
    Classify the following restaurant review as Positive, Neutral, or Negative:
    "{review_text}"
    Respond with only the sentiment word.
    """
    sentiment = ask_groq(sentiment_prompt)

    reply_prompt = f"""
    You are a polite restaurant customer service agent.
    A customer left this {sentiment} review: "{review_text}".
    Write a short and polite reply.
    """
    reply = ask_groq(reply_prompt)

    state["output"] = f"Sentiment: {sentiment}\nReply: {reply}"
    return state



# ---------- Agent 2: Sentiment Visualization ----------
def sentiment_visualization_agent(state):
    ensure_kaggle_dataset()
    user_input = state["input"]

    current_date_str = datetime.now().strftime("%Y-%m-%d")
    date_prompt = f"""
    Analyze the user's request: "{user_input}"
    Today's date is {current_date_str}.

    Your task is to extract a start and end date.
    - If the user provides a clear date range (e.g., "from 2017-05-01 to 2017-05-07"), use it.
    - If the user provides a vague request (e.g., "plot a graph", "show me last week"), determine a logical date range. For "last week", calculate the dates based on today. For a generic "plot" request, use a default range of "2022-08-01" to "2022-08-07".
    
    Respond ONLY with a valid JSON object in the format: {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}
    Do not add any other text or explanations.
    """
    
    raw_response = ask_groq(date_prompt)
    print(f"DEBUG: Raw AI response for dates: {raw_response}")

    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            date_json = json_match.group(0)
            dates = json.loads(date_json)
            start_date = dates["start_date"]
            end_date = dates["end_date"]
        else:
            raise ValueError("No JSON object found in the AI response.")

    except (ValueError, json.JSONDecodeError, KeyError) as e:
        print(f"ERROR: Failed to parse dates. Reason: {e}")
        state["output"] = "❌ Could not determine a valid date range from your request. Please try being more specific, like 'plot the data for the first week of August 2022'."
        return state

    print("⏳ Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded.")
    
    # ---Check for 'Date' (capital) and rename it to 'date' (lowercase) ---
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    
    if 'date' not in df.columns or 'Review Text' not in df.columns:
        print("DEBUG: DataFrame columns are:", df.columns)
        state["output"] = "❌ The dataset is missing a required 'date' or 'Review' column."
        return state
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    print(f"🗓️ Filtering data from {start_date} to {end_date}...")
    mask = (df['date'].dt.date >= pd.to_datetime(start_date).date()) & (df['date'].dt.date <= pd.to_datetime(end_date).date())
    filtered_df = df.loc[mask]

    if filtered_df.empty:
        state["output"] = f"ℹ️ No review data found for the period {start_date} to {end_date}."
        return state

    print("🧠 Performing sentiment analysis on filtered data...")
    sample_df = filtered_df.head(100).copy()
    
    sentiments = []
    for review in sample_df['Review Text']: 
        sentiment_prompt = f"Classify this review as Positive, Negative, or Neutral: \"{review[:500]}\". Respond with one word."
        sentiment = ask_groq(sentiment_prompt, model="llama3-8b-8192")
        sentiments.append(sentiment)

    sample_df['sentiment'] = sentiments
    
    # --- Clean up the sentiment data ---
    def clean_sentiment(sentiment):
        # Remove periods and convert to title case
        cleaned = sentiment.replace('.', '').strip().title()
        if cleaned not in ['Positive', 'Negative', 'Neutral']:
            # If it's still not a valid category, default to Neutral
            return 'Neutral'
        return cleaned

    sample_df['sentiment'] = sample_df['sentiment'].apply(clean_sentiment)

    count_df = sample_df.groupby([sample_df['date'].dt.date, 'sentiment']).size().reset_index(name='count')
    count_df.rename(columns={'date': 'day'}, inplace=True)

    print("🎨 Creating plot...")
    plt.figure(figsize=(12, 7))
    sns.barplot(data=count_df, x='day', y='count', hue='sentiment', palette={"Positive": "g", "Negative": "r", "Neutral": "b"})
    plt.xticks(rotation=45)
    plt.title(f"Sentiment Trend from {start_date} to {end_date} (Sample of {len(sample_df)} reviews)")
    plt.xlabel("Date")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    save_path = f"outputs/sentiment_plot_{start_date}_to_{end_date}.png"
    plt.savefig(save_path)
    plt.close()

    state["output"] = f"✅ Sentiment plot created successfully! It's saved here: {save_path}"
    return state


# ---------- Router Node (Keyword-based for reliability) ----------
def router(state):
    """Decides which agent to route the user's request to."""
    user_input = state["input"].lower()
    plot_keywords = ["plot", "chart", "graph", "trend", "week", "month", "year", "data", "from", "to"]

    if any(keyword in user_input for keyword in plot_keywords):
        decision = "plot"
    else:
        decision = "feedback"

    print(f"Routing decision: '{decision}'")
    state["route"] = decision
    return state


# ---------- LangGraph Setup ----------
graph = StateGraph(dict)

graph.add_node("router", router)
graph.add_node("feedback", feedback_response_agent)
graph.add_node("plot", sentiment_visualization_agent)

graph.set_entry_point("router")

# Conditional routing based on `state["route"]`
def route_condition(state):
    return state["route"]

graph.add_conditional_edges(
    "router",
    route_condition,
    {
        "feedback": "feedback",
        "plot": "plot"
    }
)



# ---------- Run System ----------
if __name__ == "__main__":
    ensure_kaggle_dataset()

    app = graph.compile()

    while True:
        user_prompt = input("\nAsk me something (or type 'exit'): ")
        if user_prompt.lower() == "exit":
            break
        result = app.invoke({"input": user_prompt})
        print("\n" + result["output"])
