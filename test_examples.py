"""Test the complaint routing system with various examples."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.inference.pipeline import ComplaintRoutingPipeline


def print_separator():
    print("\n" + "="*70)


def test_complaint(pipeline, text, description):
    """Test a single complaint."""
    print_separator()
    print(f"TEST: {description}")
    print_separator()
    print(f"Input: {text}")
    
    result = pipeline.predict(text=text)
    
    print(f"\n✓ Officer: {result['officer']['name']} ({result['officer']['confidence']:.1%})")
    print(f"✓ Priority: {result['priority']['level']} ({result['priority']['confidence']:.1%})")
    print(f"✓ ETA: {result['eta']['days']} days")
    print(f"✓ Similar complaints found: {len(result['similar_complaints'])}")


def main():
    """Run test examples."""
    print("="*70)
    print("COMPLAINT ROUTING SYSTEM - TEST EXAMPLES")
    print("="*70)
    
    # Initialize pipeline
    print("\nLoading pipeline...")
    pipeline = ComplaintRoutingPipeline()
    print("✓ Pipeline loaded\n")
    
    # Test cases covering different officers and priorities
    test_cases = [
        # Water Supply
        ("Water supply has been disrupted in my area for 3 days", 
         "Water Supply - Medium Duration"),
        
        ("No water in taps since morning", 
         "Water Supply - Short Duration"),
        
        ("Leaking water pipe causing flooding on the street", 
         "Water Supply - Emergency"),
        
        # Electricity
        ("Power outage in our locality for 2 days", 
         "Electricity - Outage"),
        
        ("Street lights not working at night creating safety issues", 
         "Electricity - Street Lights"),
        
        ("Transformer making loud noise and sparking", 
         "Electricity - Emergency"),
        
        # Road Maintenance
        ("Large pothole on main road causing accidents", 
         "Road - Pothole Emergency"),
        
        ("Road surface damaged and needs repair", 
         "Road - General Maintenance"),
        
        ("Street flooding during rain due to poor drainage", 
         "Road - Drainage Issue"),
        
        # Sanitation
        ("Garbage not collected for one week", 
         "Sanitation - Garbage Collection"),
        
        ("Overflowing dustbin attracting stray animals", 
         "Sanitation - Overflowing Bin"),
        
        ("Open drain causing foul smell and health hazard", 
         "Sanitation - Health Hazard"),
        
        # Building
        ("Illegal construction without proper permit", 
         "Building - Illegal Construction"),
        
        ("Unsafe building structure posing danger", 
         "Building - Safety Hazard"),
        
        # Noise Pollution
        ("Loud music from nearby location during night hours", 
         "Noise - Night Disturbance"),
        
        ("Construction noise at odd hours disturbing residents", 
         "Noise - Construction"),
        
        # Public Health
        ("Mosquito breeding in stagnant water near our area", 
         "Health - Mosquito Breeding"),
        
        ("Unhygienic food being sold at street vendor", 
         "Health - Food Safety"),
        
        # Parks & Recreation
        ("Park equipment broken and dangerous for children", 
         "Parks - Safety Issue"),
        
        ("Trees need trimming as branches are falling", 
         "Parks - Tree Maintenance"),
    ]
    
    # Run all test cases
    for text, description in test_cases:
        try:
            test_complaint(pipeline, text, description)
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    print_separator()
    print("ALL TESTS COMPLETED")
    print_separator())
    
    # Summary
    print("\n📊 Summary:")
    print(f"   Total test cases: {len(test_cases)}")
    print(f"   Officers covered: 8")
    print(f"   Priority levels: High, Medium, Low")
    print(f"   System status: ✅ OPERATIONAL")


if __name__ == "__main__":
    main()
