
import os
import lamini

from lamini import Lamini


# --------------------------------------------------
# 1. Configure Lamini API
# --------------------------------------------------

api_key = os.environ.get("LAMINI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "LAMINI_API_KEY environment variable is missing."
    )

lamini.api_key = api_key


# --------------------------------------------------
# 2. Select the base model
# --------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

llm = Lamini(model_name=BASE_MODEL)


# --------------------------------------------------
# 3. Prepare training data
# --------------------------------------------------

def get_training_data():

    data = [
        {
            "input": "What is Lamini?",
            "output": (
                "Lamini is a platform for building, "
                "fine-tuning and deploying large language models."
            )
        },
        {
            "input": "Is the Lamini type system similar to Python?",
            "output": (
                "Yes. Lamini uses Pydantic BaseModel "
                "for its type system."
            )
        },
        {
            "input": "Can I stop a running Lamini tuning job?",
            "output": (
                "Yes. The cancel_job() function can be "
                "used to stop a running tuning job."
            )
        },
        {
            "input": "Does Lamini require specialized hardware?",
            "output": (
                "You can use the Lamini Python SDK from a regular "
                "computer. Lamini hosts the GPU infrastructure "
                "for cloud-based model tuning."
            )
        },
        {
            "input": "Does Lamini support question answering?",
            "output": (
                "Yes. Lamini can be used to tune language models "
                "for question-answering tasks."
            )
        }
    ]

    return data


# --------------------------------------------------
# 4. Validate the training dataset
# --------------------------------------------------

def validate_training_data(data):

    if not data:
        raise ValueError("Training dataset cannot be empty.")

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            raise TypeError(
                f"Example {index} must be a dictionary."
            )

        if not item.get("input", "").strip():
            raise ValueError(
                f"Example {index} has an empty input."
            )

        if not item.get("output", "").strip():
            raise ValueError(
                f"Example {index} has an empty output."
            )

    print(f"Dataset validated: {len(data)} examples")


# --------------------------------------------------
# 5. Configure fine-tuning hyperparameters
# --------------------------------------------------

FINETUNE_ARGS = {
    "learning_rate": 1.0e-4,
    "max_steps": 50
}


# --------------------------------------------------
# 6. Start fine-tuning
# --------------------------------------------------

def start_fine_tuning():

    training_data = get_training_data()

    validate_training_data(training_data)

    try:

        result = llm.tune(
            data_or_dataset_id=training_data,
            finetune_args=FINETUNE_ARGS
        )

        print("Fine-tuning job submitted successfully.")

        print("Job ID:", result.get("job_id"))
        print("Dataset ID:", result.get("dataset_id"))

        return result

    except Exception as exc:

        print("Fine-tuning failed:", str(exc))
        raise


# --------------------------------------------------
# 7. Execute
# --------------------------------------------------

if __name__ == "__main__":

    job_result = start_fine_tuning()
