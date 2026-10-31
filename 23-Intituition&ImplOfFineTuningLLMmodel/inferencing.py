from lamini import Lamini

# Replace with the actual tuned model name
# returned or provided by your Lamini job.

tuned_llm = Lamini(
    model_name="YOUR_TUNED_MODEL_NAME"
)

response = tuned_llm.generate(
    "What is Lamini?"
)

print(response)