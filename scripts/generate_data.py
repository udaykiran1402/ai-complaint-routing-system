"""Generate synthetic training data for the complaint routing system."""

import random
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config import OFFICERS, DATA_DIR

# Complaint templates by officer type
COMPLAINT_TEMPLATES = {
    0: [  # Water Supply Officer
        "Water supply has been disrupted in {location} for {duration}",
        "No water in taps since {duration}",
        "Water pressure is very low in {location}",
        "Leaking water pipe on {location}",
        "Water contamination issue in {location}",
        "Broken water main near {location}",
        "Water supply disrupted for 3 days in {location}",
        "No water for {duration} - urgent issue",
    ],
    1: [  # Electricity Officer
        "Power outage in {location} for {duration}",
        "Frequent electricity cuts in {location}",
        "Transformer not working near {location}",
        "Street lights not functioning in {location}",
        "Voltage fluctuation causing appliance damage",
        "Electric pole damaged at {location}",
        "Power outage in {location} for 2 days - urgent",
        "No electricity for {duration} affecting daily life",
        "Transformer sparking at {location} - emergency",
    ],
    2: [  # Road Maintenance Officer
        "Large pothole on {location} causing accidents",
        "Road surface damaged at {location}",
        "Street needs urgent repair in {location}",
        "Broken pavement at {location}",
        "Road flooding during rain at {location}",
        "Traffic congestion due to bad road at {location}",
    ],
    3: [  # Sanitation Officer
        "Garbage not collected for {duration} in {location}",
        "Overflowing dustbin at {location}",
        "Drainage blocked in {location}",
        "Foul smell from garbage dump near {location}",
        "Stray animals spreading garbage at {location}",
        "Open drain causing health hazard at {location}",
    ],
    4: [  # Building Inspector
        "Illegal construction at {location}",
        "Building without proper permit at {location}",
        "Unsafe structure at {location}",
        "Unauthorized floor addition at {location}",
        "Building code violation at {location}",
        "Dangerous building condition at {location}",
    ],
    5: [  # Noise Pollution Officer
        "Loud music from {location} during night",
        "Construction noise at odd hours near {location}",
        "Industrial noise pollution from {location}",
        "Loudspeaker misuse at {location}",
        "Vehicle horn noise at {location}",
        "Continuous noise disturbance from {location}",
    ],
    6: [  # Public Health Officer
        "Mosquito breeding near {location}",
        "Unhygienic conditions at {location}",
        "Disease outbreak in {location}",
        "Contaminated food being sold at {location}",
        "Public toilet in poor condition at {location}",
        "Health hazard from {location}",
    ],
    7: [  # Parks & Recreation Officer
        "Park maintenance needed at {location}",
        "Broken playground equipment at {location}",
        "Trees need trimming at {location}",
        "Park lighting not working at {location}",
        "Garden in poor condition at {location}",
        "Vandalism in park at {location}",
    ],
}

LOCATIONS = [
    "Main Street", "Park Avenue", "Gandhi Road", "MG Road", "Station Road",
    "Market Area", "Residential Colony", "Industrial Area", "City Center",
    "Old Town", "New Layout", "Sector 5", "Block A", "Zone 3"
]

DURATIONS = [
    "2 days", "3 days", "1 week", "several days", "many days",
    "the past week", "a long time"
]

# Priority mapping based on keywords
HIGH_PRIORITY_KEYWORDS = ["urgent", "emergency", "accident", "health hazard", "dangerous", "outbreak", "sparking", "fire", "flooding", "contaminated"]
MEDIUM_PRIORITY_KEYWORDS = ["broken", "damaged", "not working", "poor condition", "disrupted", "outage", "leaking"]
LOW_PRIORITY_KEYWORDS = ["maintenance", "needs", "trimming", "cleaning"]

# Duration keywords that affect priority
LONG_DURATION_KEYWORDS = ["week", "weeks", "many days", "several days", "long time"]
SHORT_DURATION_KEYWORDS = ["today", "morning", "hours", "1 day"]


def generate_complaint(officer_id: int) -> dict:
    """Generate a single complaint."""
    template = random.choice(COMPLAINT_TEMPLATES[officer_id])
    location = random.choice(LOCATIONS)
    duration = random.choice(DURATIONS)
    
    text = template.format(location=location, duration=duration)
    
    # Determine priority based on keywords AND duration
    text_lower = text.lower()
    
    # Start with base priority
    if any(kw in text_lower for kw in HIGH_PRIORITY_KEYWORDS):
        priority = 2  # High
        eta_days = random.uniform(0.5, 3)
    elif any(kw in text_lower for kw in MEDIUM_PRIORITY_KEYWORDS):
        priority = 1  # Medium
        eta_days = random.uniform(2, 7)
    else:
        priority = 0  # Low
        eta_days = random.uniform(5, 14)
    
    # Adjust priority based on duration mentioned
    if any(dur in text_lower for dur in LONG_DURATION_KEYWORDS):
        # Long duration increases priority
        if priority == 0:
            priority = 1  # Low -> Medium
        elif priority == 1:
            priority = 2  # Medium -> High
        eta_days = max(1, eta_days - 2)  # Reduce ETA for urgent cases
    
    # Special cases for specific complaint types
    if officer_id == 1:  # Electricity
        # Power outages are more urgent
        if "outage" in text_lower or "power" in text_lower:
            if "2 days" in text_lower or "3 days" in text_lower or any(dur in text_lower for dur in LONG_DURATION_KEYWORDS):
                priority = max(1, priority)  # At least Medium
                eta_days = random.uniform(1, 4)
    
    if officer_id == 0:  # Water Supply
        # Water disruption is critical
        if "disrupted" in text_lower or "no water" in text_lower:
            if any(dur in text_lower for dur in LONG_DURATION_KEYWORDS) or "days" in text_lower:
                priority = max(1, priority)  # At least Medium
                eta_days = random.uniform(1, 5)
    
    if officer_id == 2:  # Road
        # Accidents make it high priority
        if "accident" in text_lower:
            priority = 2
            eta_days = random.uniform(1, 3)
    
    # Add some randomness but keep it reasonable
    eta_days += random.uniform(-0.5, 0.5)
    eta_days = max(0.5, min(eta_days, 15))  # Cap between 0.5 and 15 days
    
    return {
        "text": text,
        "officer_id": officer_id,
        "priority": priority,
        "eta_days": round(eta_days, 1),
        "status": "resolved"
    }


def generate_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic dataset."""
    print(f"Generating {n_samples} synthetic complaints...")
    
    complaints = []
    for _ in range(n_samples):
        officer_id = random.randint(0, len(OFFICERS) - 1)
        complaint = generate_complaint(officer_id)
        complaints.append(complaint)
    
    df = pd.DataFrame(complaints)
    
    # Add complaint IDs
    df['complaint_id'] = range(1, len(df) + 1)
    
    # Reorder columns
    df = df[['complaint_id', 'text', 'officer_id', 'priority', 'eta_days', 'status']]
    
    return df


def main():
    """Generate and save training data."""
    # Generate training data
    train_df = generate_dataset(n_samples=1000)
    
    # Generate test data
    test_df = generate_dataset(n_samples=200)
    
    # Create data directory
    data_dir = DATA_DIR / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    train_path = data_dir / "train_complaints.csv"
    test_path = data_dir / "test_complaints.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nDataset generated successfully!")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"\nSaved to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    
    # Print statistics
    print(f"\nOfficer distribution:")
    for officer_id, count in train_df['officer_id'].value_counts().sort_index().items():
        print(f"  Officer {officer_id} ({OFFICERS[officer_id]['name']}): {count}")
    
    print(f"\nPriority distribution:")
    priority_map = {0: "Low", 1: "Medium", 2: "High"}
    for priority, count in train_df['priority'].value_counts().sort_index().items():
        print(f"  {priority_map[priority]}: {count}")
    
    print(f"\nETA statistics:")
    print(f"  Mean: {train_df['eta_days'].mean():.1f} days")
    print(f"  Min: {train_df['eta_days'].min():.1f} days")
    print(f"  Max: {train_df['eta_days'].max():.1f} days")


if __name__ == "__main__":
    main()
