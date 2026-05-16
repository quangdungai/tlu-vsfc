import sys
sys.path.insert(0, '.')
from utils.absa_pipeline import run_absa_pipeline
from utils.model_loader import get_model_and_tokenizer

model, tokenizer = get_model_and_tokenizer()
test_cases = [
    "Co day nhiet tinh nhung phong hoi nong, chuong trinh dao tao tot",
    "Giang vien tuyet voi, tai lieu qua cu",
    "Phong hoc mat me nhung thay kho tinh",
]
for text in test_cases:
    print(f"\nInput: {text}")
    results = run_absa_pipeline(text, model, tokenizer)
    for r in results:
        aspect = r["aspect"]
        label = r["sentiment"]["label"]
        conf = r["sentiment"]["confidence"]
        print(f"  -> {aspect}: {label} ({conf:.1%})")
print("\nPipeline test PASSED")
