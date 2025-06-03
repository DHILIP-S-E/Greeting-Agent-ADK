import datetime
import os
import logging
import sys # For graceful exit

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Google ADK specific imports
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
# Using the import path from the google-generativeai library for safety settings
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Libraries for external data retrieval
import requests
from googleapiclient.discovery import build
import googleapiclient.errors # Import for specific error handling

# Configure basic logging for ADK events and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Retrieve Environment Variables ---
# ADK automatically picks up GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, etc.,
# but explicitly getting API keys for specific tools (like CSE) is necessary.
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# --- Initial Environment Variable Check ---
if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
    logging.error("Missing GOOGLE_CSE_API_KEY or GOOGLE_CSE_ID in your .env file. "
                  "Please ensure these are set for the Google Custom Search tool to function.")
    sys.exit(1) # Exit if critical variables are missing

# --- 3. Tool Definition: Your External Data Connector ---
def get_special_day_info_from_external_source(date_query: str) -> dict:
    """
    TOOL: This function retrieves real-time information about special or
    international days for a given date from an external source (Google Custom Search).
    It is designed to be called by the ADK agent.

    Args:
        date_query (str): The date string (e.g., "May 21") that the LLM will
                          provide to this tool.

    Returns:
        dict: A dictionary where keys are day titles and values are their descriptions.
              Returns an empty dict if no special days are found or if an API error occurs.
              In case of API error, it logs the error but still returns an empty dict
              so the agent can gracefully respond that it couldn't find information.
    """
    logging.info(f"TOOL CALL: 'get_special_day_info_from_external_source' called with query: '{date_query}'")
    results = {}

    try:
        results = _search_special_days(date_query)
    except Exception as e:
        # Catch any exception from _search_special_days and log it.
        # Returning an empty dict indicates no info could be retrieved.
        logging.error(f"Error fetching special day info via tool for '{date_query}': {e}")
        pass # The error is logged, and results remains empty.

    return results

def _search_special_days(date_query: str) -> dict:
    """
    Helper function to retrieve special day information from Google Custom Search API.
    Raises exceptions on API errors, which are caught by the calling tool function.
    """
    results = {}
    logging.info("Attempting to use Google Custom Search API for special days...")
    service = build("customsearch", "v1", developerKey=GOOGLE_CSE_API_KEY)
    # Refined search query for better precision
    search_query = f"international OR world OR national day {date_query} official observances"
    logging.info(f"{search_query=}")

    res = service.cse().list(q=search_query, cx=GOOGLE_CSE_ID, num=5).execute()

    if 'items' in res:
        for item in res['items']:
            title = item.get('title')
            snippet = item.get('snippet')
            # Basic filtering to ensure relevant results, focusing on 'day' in title
            if title and snippet and ("day" in title.lower() or "observance" in title.lower()):
                # Try to extract a concise description, handling potential list formats in snippets
                description = snippet.split('...')[0].strip()
                if title not in results:
                    results[title] = description
    logging.info(f"Google Custom Search API returned {len(results)} results.")
    return results

# Register your special day search function as an ADK tool
# CORRECTED: Removed 'name' and 'description' keyword arguments as FunctionTool doesn't accept them.
special_day_search_tool = FunctionTool(get_special_day_info_from_external_source)

# --- 4. Agent Definition ---
class SpecialDayAgent:
    def __init__(self, model_name: str = "gemini-pro"):
        """
        Initializes the ADK LlmAgent, setting up its model, instructions, and tools.

        Args:
            model_name (str): The Google Generative AI model to use via Vertex AI.
        """
        self.agent = LlmAgent(
            name="special_day_agent",
            model=model_name,
            description="A helpful assistant that identifies special days using a tool and generates creative messages.",
            instruction="""
            Your primary task is to identify if today is a special occasion (international, national, or awareness day).
            **First, you MUST use the 'get_special_day_info_from_external_source' tool.**
            To do this, use the current date provided in the prompt (e.g., 'May 21') and pass it as the `date_query` argument to the tool.

            Based on the tool's results:
            - If special days are found:
                - For each special day, clearly state its **Title** and a brief **Description**.
                - Then, craft a warm, fresh, and inspiring message about the day's theme.
                - Finally, create a simple, concise visual representation (e.g., emojis or a short symbolic phrase) of the day with appropriate colors and symbols.
            - If the tool reports no special days (returns an empty dictionary or an error was logged during tool execution), simply state that no widely recognized special or international days are observed today based on your current information.

            Keep your messages meaningful and your visual representations simple and relevant.
            """,
            tools=[special_day_search_tool],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARMS_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARMS_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARMS_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        self.runner = Runner(self.agent)

    def run_daily_check(self):
        """
        Executes the ADK agent's daily check. The agent will orchestrate its own
        tool calls based on its instructions.
        """
        current_date_obj = datetime.date.today()
        # Explicitly provide the "Month Day" format for the tool call guidance
        current_date_formatted = current_date_obj.strftime('%B %d') # e.g., "May 21"
        initial_prompt = (f"Hello! Today's full date is {current_date_obj.strftime('%B %d, %Y')}. "
                          f"The current 'Month Day' for tool queries is '{current_date_formatted}'. "
                          "Please check for special or international days and generate greetings.")

        print(f"\n--- Running ADK Special Day Agent for {current_date_obj.strftime('%B %d, %Y')} ---")
        logging.info(f"Agent's initial prompt: {initial_prompt}")

        try:
            for event in self.runner.run(initial_prompt):
                if event.type == "tool_code":
                    logging.info(f"ADK Event: Agent called tool: {event.tool_code.tool_name} with args: {event.tool_code.args}")
                    print(f"DEBUG: Agent decided to use tool: {event.tool_code.tool_name} with arguments {event.tool_code.args}")
                elif event.type == "tool_response":
                    logging.info(f"ADK Event: Tool response: {event.tool_response.output}")
                    print(f"DEBUG: Tool returned: {event.tool_response.output}")
                elif event.type == "agent_response":
                    print("\n--- Agent's Final Output ---")
                    print(event.agent_response.text)
                    break
                elif event.type == "error":
                    logging.error(f"ADK Event: An error occurred during agent execution: {event.error.message}")
                    print(f"Agent encountered an error: {event.error.message}")
                    break
        except Exception as e:
            logging.exception("Unhandled error during ADK agent run.")
            print(f"An unhandled error occurred during ADK agent run: {e}")

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # Ensure you've followed ALL the setup steps in the README.md:
    # 1. Install Python libraries: pip install dotenv google-adk google-api-python-client google-generativeai
    # 2. Configure Google Cloud Project and enable APIs (Vertex AI, Custom Search API).
    # 3. Authenticate with gcloud auth application-default login.
    # 4. Set environment variables in your .env file.
    # 5. IMPORTANT: Your 'get_special_day_info_from_external_source' tool
    #    now uses Google Custom Search API. Ensure GOOGLE_CSE_API_KEY
    #    and GOOGLE_CSE_ID are correctly set in your .env.

    special_day_agent_instance = SpecialDayAgent(model_name="gemini-pro")
    special_day_agent_instance.run_daily_check()

    print("\n--- ADK Agent execution complete. ---")