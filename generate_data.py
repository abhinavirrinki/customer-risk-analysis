import random

# Configuration
NUM_CUSTOMERS = 100
MAX_RECORDS_PER_CUSTOMER = 5
OUTPUT_FILE = "generated_sample_data.py"

# Data generation pools
BEHAVIORS = [
    "Customer cancelled ride 2 minutes before pickup.",
    "Customer disputed a charge claiming they were never picked up.",
    "Customer left a 1-star rating citing rude behavior from driver.",
    "Customer requested refund for a delayed delivery.",
    "Customer repeatedly cancels after driver is assigned.",
    "Customer filed complaint about food quality, requested full refund.",
    "Customer has clean history, no complaints in past 6 months.",
    "Customer used promo code fraudulently across multiple accounts.",
    "Customer disputed multiple charges in one week.",
    "Customer complained driver took a longer route, requested partial refund.",
    "Customer cancelled last minute due to a genuine emergency.",
    "Customer account flagged for unusual login locations.",
    "Customer completed 50+ rides with 5-star ratings.",
    "Customer reported a missing item in their food delivery.",
    "Customer payment method failed 3 times consecutively before succeeding.",
    "Customer requested a chargeback directly through their bank.",
    "System detected possible GPS spoofing during the ride.",
    "Customer tipped the driver 20% and left a positive review."
]

def generate_records():
    cases = []
    for i in range(1, NUM_CUSTOMERS + 1):
        customer_id = f"C{i:03d}"
        
        # Determine if this is a "good", "bad", or "mixed" customer to group behaviors
        profile_type = random.choices(["good", "risky", "mixed"], weights=[0.5, 0.3, 0.2])[0]
        num_records = random.randint(1, MAX_RECORDS_PER_CUSTOMER)
        
        for _ in range(num_records):
            if profile_type == "good":
                text = random.choice([b for b in BEHAVIORS if "clean" in b or "positive" in b or "completed" in b or "emergency" in b])
            elif profile_type == "risky":
                text = random.choice([b for b in BEHAVIORS if "disputed" in b or "fraud" in b or "chargeback" in b or "spoofing" in b or "unusual" in b])
            else:
                text = random.choice(BEHAVIORS)
                
            cases.append({"customer_id": customer_id, "text": text})
            
    return cases

def main():
    cases = generate_records()
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("SAMPLE_CASES = [\n")
        for case in cases:
            f.write(f'    {{"customer_id": "{case["customer_id"]}", "text": "{case["text"]}"}},\n')
        f.write("]\n")
        
    print(f"Successfully generated {len(cases)} records for {NUM_CUSTOMERS} customers.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()