from flask import Flask, request, jsonify
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    # Get user input
    data = request.json
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Encode input and generate a response
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the bot's reply
    bot_reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"reply": bot_reply})

if __name__ == '__main__':
    app.run(debug=True)

# app = Flask(__name__)

# # Load a pretrained chatbot model from Hugging Face
# chatbot = pipeline('text-generation', model='gpt2')

# @app.route('/')
# def home():
#     return "Chatbot is running!"

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('message', '')
#     if not user_input:
#         return jsonify({"error": "No input provided"}), 400
    
#     response = chatbot(user_input, max_length=50, num_return_sequences=1,temperature=0.7,top_k=50, top_p=0.9)
#     bot_reply = response[0]['generated_text']

#     return jsonify({"reply": bot_reply})

# if __name__ == '__main__':
#     app.run(debug=True)
