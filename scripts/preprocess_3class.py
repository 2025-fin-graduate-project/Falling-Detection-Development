#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Add label_3class to dataset based on VHSSC rule.")
    parser.add_argument("--input", default="dataset/final_dataset.csv", help="Input CSV path")
    parser.add_argument("--output", default="dataset/final_dataset_3class.csv", help="Output CSV path")
    parser.add_argument("--threshold-low", type=float, default=0.07, help="VHSSC threshold for Almost Fall")
    parser.add_argument("--threshold-high", type=float, default=0.2, help="VHSSC threshold for Fall")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    print("Applying 3-class labeling rules...")
    # Default is non_fall (0)
    df['label_3class'] = 0

    # Rule for Almost Fall (1):
    # - If label is 0 but VHSSC is high (pre-fall or stumble)
    # - If label is 1 but VHSSC hasn't reached impact velocity yet
    almost_fall_mask = (
        ((df['label'] == 0) & (df['VHSSC'] >= args.threshold_low) & (df['VHSSC'] < args.threshold_high)) |
        ((df['label'] == 1) & (df['VHSSC'] < args.threshold_high))
    )
    df.loc[almost_fall_mask, 'label_3class'] = 1

    # Rule for Fall (2):
    # - If label is 1 and VHSSC is very high (impact/descent)
    # - We also include label=0 with very high VHSSC if it's likely a continuation of a fall
    #   (Though usually label=1 should cover the main fall)
    fall_mask = (df['label'] == 1) & (df['VHSSC'] >= args.threshold_high)
    df.loc[fall_mask, 'label_3class'] = 2
    
    # Optional: If label=0 and VHSSC is very high, it might be noise or post-fall.
    # In many cases, it's safer to keep it as non-fall (0) or almost-fall (1) 
    # to avoid false positives during normal high-speed activities.
    # But let's check if we should map it to 1 or 2.
    # Given the previous observation, frame 291 had VHSSC=0.64 and label=0.
    # If we map this to 'Fall', it might be okay since the person just fell.
    # But if we want to be strict about 'intermediate stage', we focus on the transition.
    
    # Let's see the distribution after this
    print("Label Distribution:")
    print(df['label_3class'].value_counts().sort_index())

    print(f"Saving to {args.output}...")
    df.to_csv(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
