Fine-tuning means taking an already-trained model and training it further on your own carefully prepared examples so that it learns a particular task, behavior, or domain-specific pattern.
----
A practical example
Suppose you have a pretrained Llama model.
You ask:
Question:
"What is Lamini?"
The original model might respond with a generic explanation.

Now you provide this training example:
{    "input": "What is Lamini?",    "output": "Lamini is a platform for building, fine-tuning and deploying LLMs."}


You provide many high-quality examples like this to the model.
During fine-tuning, the model adjusts its internal weights to become better at producing the desired answers.
After training, when you ask:
"What is Lamini?"
It should become more likely to generate the answer you demonstrated.

Important: Fine-tuning does not work like adding a new row to a database. It changes the model's learned parameters. It also does not guarantee that the model will memorize every training answer perfectly.
--------------------

What is quantization? How does it work?
This is especially useful when you want to run LLMs locally using Ollama on your MacBook.
Imagine you have a pretrained Llama model.
The model contains billions of numerical weights.
Normally, these weights might be stored using floating-point numbers.
For example:
0.12345678
-0.98765432
0.76543210
Each number occupies memory.
Quantization means representing these numerical weights using lower-precision data types.
For example:
- FP32: 32 bits per weight.
- FP16: 16 bits per weight.
- BF16: 16 bits per weight.
- INT8: 8 bits per weight.
- INT4: 4 bits per weight.
The basic idea is to reduce the number of bits required to represent each weight.
---------------------
Fine-tuning vs Quantization
This distinction is extremely important.
Fine-tuning	                                            Quantization
Changes trainable model weights	                        Changes numerical representation
Learns specific behaviors	                            Reduces model storage/memory
Requires training	                                    Can be done after training
Can improve task performance	                        Can introduce some accuracy loss
Example: LoRA	                                        Example: INT4

You can even combine them!
What is QLoRA?
QLoRA combines quantization and LoRA.

Its basic approach:
1. Load the original model in a low-bit quantized representation, commonly 4-bit.
2. Keep the quantized base model frozen.
3. Add trainable LoRA adapters.
4. Train the adapters.
5. Use the resulting adapters with the base model for inference.
QLoRA makes fine-tuning large models more accessible on hardware with limited GPU memory.