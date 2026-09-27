#!/usr/bin/env python3
"""
Main Entry Point

Usage:
    python3 src/main.py \
        --train-dir dataset/train \
        --test-dir dataset/test \
        --output-dir output \
        --model-dir models
"""

import argparse
import os
from pipeline import train_pipeline, predict_on_test, write_output


def main():
    parser = argparse.ArgumentParser(
        description="ML Challenge 2026: Business Entity Resolution"
    )
    parser.add_argument(
        "--train-dir",
        default="dataset/train",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Path to output directory for TSV results",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Path to save/load trained model",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; load model from disk (for inference only)",
    )
    
    args = parser.parse_args()
    
    # Paths
    train_source1 = os.path.join(args.train_dir, "train_source1.tsv")
    train_source2 = os.path.join(args.train_dir, "train_source2.tsv")
    train_source3 = os.path.join(args.train_dir, "train_source3.tsv")
    train_gt = os.path.join(args.train_dir, "train_ground_truth.tsv")
    
    test_source1 = os.path.join(args.test_dir, "test_source1.tsv")
    test_source2 = os.path.join(args.test_dir, "test_source2.tsv")
    test_source3 = os.path.join(args.test_dir, "test_source3.tsv")
    
    model_path = os.path.join(args.model_dir, "model")
    
    # Train
    if not args.skip_train:
        print("=" * 70)
        print("TRAINING PHASE")
        print("=" * 70)
        model = train_pipeline(
            train_source1, train_source2, train_source3, train_gt, model_path
        )
    else:
        print("Loading pre-trained model...")
        from model import MatchingModel
        model = MatchingModel()
        model.load(model_path)
        print(f"Model loaded from {model_path}")
    
    # Predict
    print("\n" + "=" * 70)
    print("INFERENCE PHASE")
    print("=" * 70)
    matching_results, candidate_pairs = predict_on_test(
        test_source1, test_source2, test_source3, model, args.output_dir
    )
    
    # Write output
    print("\n" + "=" * 70)
    print("WRITING OUTPUT")
    print("=" * 70)
    write_output(matching_results, candidate_pairs, args.output_dir)
    
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"Run: python3 utils/validate_submission.py \\")
    print(f"    --matching {args.output_dir}/matching_results.tsv \\")
    print(f"    --candidate {args.output_dir}/candidate_pairs.tsv \\")
    print(f"    --test-dir {args.test_dir}")
    
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
