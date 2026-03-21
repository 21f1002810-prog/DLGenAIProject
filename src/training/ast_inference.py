import torch
import pandas as pd
from pathlib import Path
import librosa

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

GENRES = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_file(model, feature_extractor, file_path):

    audio, _ = librosa.load(file_path, sr=16000)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_values = inputs["input_values"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values)

    pred = torch.argmax(outputs.logits, dim=1).item()

    return GENRES[pred]


def main():

    project_root = Path(__file__).resolve().parents[2]

    test_csv = project_root / "dataset" / "test.csv"
    mashup_dir = project_root / "dataset" / "mashups"

    df = pd.read_csv(test_csv)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    model = AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=10
    )

    model.load_state_dict(torch.load("ast_best.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    predictions = []

    for _, row in df.iterrows():

        file_path = mashup_dir / row["id"]

        pred = predict_file(model, feature_extractor, file_path)

        predictions.append(pred)

    df["genre"] = predictions

    df[["id","genre"]].to_csv("submission.csv", index=False)

    print("Submission file saved!")


if __name__ == "__main__":
    main()