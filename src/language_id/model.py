# model.py
from transformers import AutoModelForSequenceClassification

def build_model(
    model_name: str,
    num_labels: int,
    id2label: dict[str, str],
    label2id: dict[str, int],
):
    """
    Returns a Hugging Face classification model with label mappings.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model
