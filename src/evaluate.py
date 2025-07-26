"""
evaluate.py - Evaluation and submission creation for test data.
"""
import torch
import numpy as np
from utils import create_submission

def evaluate_and_submit(model, test_paths, submission_path):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for sample_path in tqdm(test_paths, desc="Creating Submission"):
            sample_id = os.path.basename(sample_path)

            input_stack = []
            for src_id in [1, 75, 150, 225, 300]:
                src_file = os.path.join(sample_path, f"receiver_data_src_{src_id}.npy")
                data = np.load(src_file).astype(np.float32)
                data = data[::3]  # Downsample time axis
                input_stack.append(data)
            input_tensor = np.stack(input_stack, axis=0)  # Shape: (5, ~250, 31)
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

            output = model(input_tensor)
            output = nn.functional.interpolate(output, size=(300, 1259), mode="bilinear", align_corners=False)
            prediction = output.squeeze().cpu().numpy().astype(np.float64)

            create_submission(sample_id, prediction, submission_path)