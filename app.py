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
