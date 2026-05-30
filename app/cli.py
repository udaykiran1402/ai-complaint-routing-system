"""Command-line interface for complaint routing system."""

import argparse
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.pipeline import ComplaintRoutingPipeline


def print_result(result: dict):
    """Pretty print the prediction result."""
    print("\n" + "="*70)
    print("COMPLAINT ROUTING RESULT")
    print("="*70)
    
    print(f"\n📝 Input Type: {result['input_type'].upper()}")
    print(f"\n💬 Complaint Text:")
    print(f"   {result['complaint_text']}")
    
    print(f"\n👤 Assigned Officer:")
    print(f"   {result['officer']['name']}")
    print(f"   Confidence: {result['officer']['confidence']:.1%}")
    
    print(f"\n⚠️  Priority Level:")
    print(f"   {result['priority']['level']}")
    print(f"   Confidence: {result['priority']['confidence']:.1%}")
    
    print(f"\n⏱️  Estimated Resolution Time:")
    print(f"   {result['eta']['days']} days ({result['eta']['description']})")
    
    print(f"\n🔍 Similar Past Complaints:")
    if result['similar_complaints']:
        for i, complaint in enumerate(result['similar_complaints'], 1):
            print(f"\n   {i}. Similarity: {complaint['similarity_score']:.2%}")
            print(f"      Text: {complaint['text'][:80]}...")
            print(f"      Status: {complaint['status']}")
    else:
        print("   No similar complaints found")
    
    print("\n" + "="*70)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Complaint Auto-Routing System - Process and route complaints"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t",
        type=str,
        help="Text complaint"
    )
    input_group.add_argument(
        "--audio", "-a",
        type=str,
        help="Path to audio file"
    )
    input_group.add_argument(
        "--video", "-v",
        type=str,
        help="Path to video file"
    )
    
    # Output options
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save result to file"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        print("Loading complaint routing system...")
        pipeline = ComplaintRoutingPipeline()
        
        # Process complaint
        print("\nProcessing complaint...")
        result = pipeline.predict(
            text=args.text,
            audio_path=args.audio,
            video_path=args.video
        )
        
        # Output result
        if args.json:
            output = json.dumps(result, indent=2)
            print(output)
        else:
            print_result(result)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Result saved to {args.output}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Generated training data: python scripts/generate_data.py")
        print("  2. Trained models: python scripts/train_pipeline.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
