import asyncio
import logging
import os

import dotenv
from plugins.lights_plugin import LightsPlugin
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory


async def main():
    dotenv.load_dotenv()

    # Initialize the kernel
    kernel = Kernel()

    OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
    OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")

    # Retrieve the chat completion service for later use.
    chat_completion = AzureChatCompletion(
        deployment_name=OPENAI_DEPLOYMENT_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    # Add an Azure OpneAI chat completion service to the karnel builder.
    kernel.add_service(chat_completion)

    # Add the logging service to the karnel to help debug the AI agent.
    logging.basicConfig(
        format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    # With plugins, can give the AI agent the ability to run code to retrieve information from external sources or to perform actions.
    # Add a plugin that allows the AI agent to interact with a light bulb.
    kernel.add_plugin(
        LightsPlugin(),
        plugin_name="Lights",
    )

    """
    Semantic Kernel leverages function calling to provide planning.

    To enable automatic function calling,
    we need to create appropriate execution settings so that Semantic Kernel knows to automatically invoke the functions in the kernel when the AI agent requests them.
    """
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create a history of the conversation
    history = ChatHistory()

    # Initiate a back-and-forth chat
    userInput = None
    while True:
        # Collect user input
        userInput = input("User > ")

        # Terminate the loop if the user says "exit"

        if userInput == "exit":
            break

        # Add user input to the history
        history.add_user_message(userInput)

        # Get the response from the AI
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Print the results
        print("Assistant > " + str(result))
        print(LightsPlugin.lights)

        # Add the message from the agent to the chat history
        history.add_message(result)


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
