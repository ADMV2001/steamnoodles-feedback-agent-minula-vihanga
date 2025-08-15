# SteamNoodles Automated Feedback Agent

This project is an automated, multi-agent system designed to handle customer feedback for the SteamNoodles restaurant chain. It uses a large language model (LLM) powered by the Groq API to understand and process user requests, routing them to the appropriate agent for a response or data visualization.

    Name: A.D.Minula Vihanga
    University: NSBM Green  University
    Year: 2nd Year

# Summary of Approach:

This system is built on a multi-agent architecture using Python and the LangGraph framework. This approach allows for a clear separation of concerns and a logical flow of control.

The system consists of three main components:

Router Agent: This is the entry point for all user requests. It uses the Groq LLM (llama3-8b-8192) to analyze the user's prompt and determine their intent: either to get a reply for a customer review or to visualize sentiment data. It then routes the request to the appropriate agent. For reliability, this router uses a keyword-based approach.

Feedback Response Agent: This agent handles individual customer reviews. It performs a two-step process:

->First, it asks the LLM to classify the sentiment of the review as Positive, Negative, or Neutral.

->Second, using the original review and the determined sentiment, it prompts the LLM again to generate a short, polite, and context-aware reply suitable for a customer service representative.

Sentiment Visualization Agent: This agent generates plots of sentiment trends over time.

->It uses the LLM to extract a start and end date from the user's natural language request.

->It then uses the Pandas library to load and filter the Yelp restaurant review dataset from Kaggle.

->For the reviews within the specified date range, it performs on-the-fly sentiment analysis.

->Finally, it uses Seaborn and Matplotlib to create a bar chart of the sentiment counts per day and saves it as a .png image in the outputs directory.

->The entire system is powered by the Groq API, which provides extremely fast inference speeds, making the agent's responses feel instantaneous.


# Setup and Installation:
Follow these steps to get the project running on your local machine.

# Prerequisites
      Python 3.8 or newer
      Git

1. Clone the Repository
   
       git clone [GitHub Link]

2. Create a Virtual Environment
  It's highly recommended to use a virtual environment to manage project dependencies.

  For Windows:
    
      python -m venv venv
      venv\Scripts\activate

  # if can not activate, run this and activate after that
    
      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

  For macOS/Linux:
    
      python3 -m venv venv
      source venv/bin/activate

3. Install Dependencies
  Install all the required Python libraries from the requirements.txt file.

      pip install -r requirements.txt

(Note: If a requirements.txt file is not provided, you can install the packages manually: pip install python-dotenv pandas seaborn matplotlib groq langgraph kaggle)

4. Set Up Environment Variables
  You will need a Groq API key to run this project.
  Create a file named .env in the root of the project directory.

   Add your Groq API key to the file like this:

          GROQ_API_KEY="your_actual_api_key_here"

5. Kaggle API Credentials
  The script will automatically download the required dataset from Kaggle. To do this, you need to have your Kaggle API credentials set up.

  Log in to your Kaggle account.
  Go to Account -> API -> Create New API Token. This will download a kaggle.json file.
  Place this kaggle.json file in the appropriate directory:

      Windows: C:\Users\<Your-Username>\.kaggle\
      macOS/Linux: ~/.kaggle/

# Dataset

This project uses the [Yelp Restaurant Reviews](https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews) dataset from Kaggle.
The dataset is downloaded automatically when the script is first run. Please ensure your Kaggle API credentials are set up as described in the setup instructions.

# Instructions to Test Both Agents
Once the setup is complete, you can run the main script from your terminal.

    python feedback_agent.py

The script will first ensure the dataset is downloaded and then present you with a prompt to interact with the agents.

# Agent 1: Testing the Feedback Response Agent
This agent is designed to handle customer reviews. Type a review and press Enter.

* Example Prompts:

        For a positive review:
        The food was absolutely wonderful, and the service was even better!

        For a negative review:
        I was very disappointed with my meal. The pasta was cold and the waiter was rude.

        For a neutral or mixed review:
        The ambiance was nice, but the portion sizes were a bit small for the price.

# Agent 2: Testing the Sentiment Visualization Agent
This agent is designed to understand requests for plots and charts over a specific time period.

* Example Prompts:

        For a specific date range:
        plot a graph from 2022 june 25 to july 25

        For a more general request:
        show me the sentiment data for the first week of august 2022

        For a vague request (will use a default date range):
        can you plot a chart for me?

▶ Expected Output:
After running a plot command, the agent will perform the analysis and save a .png file. The output in the terminal will look like this:

✅ Sentiment plot created successfully! It's saved here: outputs/sentiment_plot_2022-06-25_to_2022-07-25.png

You can then open the outputs folder in the project directory to view the generated image.
