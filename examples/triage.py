"""Triage a support ticket: three questions, one forward pass."""
import json
import tinyjev

agent = tinyjev.load("TinyJev-0.6B")

result = agent.predict({
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "questions": {
        "team":     {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                  "shipping": "Delivery status, delays, lost packages",
                                  "billing":  "Charges, invoices, payment problems"}},
        "escalate": {"type": "noul",   "instructions": "Does this need urgent human attention?"},
        "anger":    {"type": "score",  "instructions": "How angry is the customer?",
                     "criteria": ["calm", "frustrated", "very angry"]},
    }})

print(json.dumps(result["states"][0]["answers"], indent=2))
print(f"\n{result['execution']['model_ms']:.0f} ms, {result['execution']['questions']} questions, one forward pass")
