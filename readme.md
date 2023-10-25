# Chatbot Project README

This repository contains a Chatbot project that provides a friendly, AI-powered conversational experience. The project consists of several files to enable its functionality, and this README will guide you through its components and how to use them.

## Project Structure

The Chatbot project is organized into several files, each serving a specific purpose:

### `intents.json`

This JSON file defines the intents and responses used by the Chatbot. Each intent includes a set of patterns that the Chatbot can recognize and corresponding responses. For example, the "greetings" intent might include patterns like "Hello" and responses like "Hi there! How can I help you today?" You can customize these intents to create a tailored conversation experience.

### `training.py`

This Python script is responsible for training the Chatbot. It uses the `intents.json` file to generate a model capable of recognizing patterns and selecting appropriate responses. The training data is preprocessed, and a neural network model is built using TensorFlow. The trained model is then saved as `chatbot_model.h5`.

### `chatbot.py`

This script allows you to interact with the trained Chatbot. You can input messages, and the Chatbot will respond with appropriate answers based on the patterns and responses defined in `intents.json`. The model loaded from `chatbot_model.h5` is used to predict responses.

### `chatbot_GUI.py`

This is a graphical user interface (GUI) for the Chatbot, providing a user-friendly way to chat with the Chatbot. It uses the same model and logic as `chatbot.py` but presents the conversation in a visually appealing manner.

## Usage

### Training the Chatbot

1. Ensure you have the required dependencies installed. You can use `pip install -r requirements.txt` to install them.

2. Customize the intents and responses in the `intents.json` file to match the conversation you want the Chatbot to have.

3. Run `training.py` to train the Chatbot. This script processes the data, builds a neural network model, and saves it as `chatbot_model.h5`.

### Command-Line Chatbot

1. Run `chatbot.py` to interact with the Chatbot via the command line. Input your messages, and the Chatbot will respond.

### GUI Chatbot

1. Run `chatbot_GUI.py` to launch the Chatbot GUI. You'll see a text box to input your messages and a chat log for the conversation. Press "Enter" or click the "Send" button to send your message. The Chatbot's responses will appear in the chat log.

## Customization

You can tailor the Chatbot's conversation by modifying the `intents.json` file. Add or edit intents and responses to create a unique and engaging experience for users.

## Author

This Chatbot project is authored by Mustafa Ghoneim. Feel free to reach out if you have any questions or feedback.

Happy Chatting! 🤖💬
