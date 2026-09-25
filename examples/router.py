"""Decide locally whether a request needs the expensive model."""
import tinyjev

agent = tinyjev.load("TinyJev-0.6B")

REQUESTS = [
    "What's 15% of 240?",
    "Refactor this service to use dependency injection and update the tests.",
    "Translate 'good morning' into French.",
    "Find the race condition in our booking flow and propose a fix.",
]

saved = 0
for text in REQUESTS:
    a = agent.predict({"state": text, "questions": {
        "difficulty": {"type": "score", "instructions": "How hard is this request?",
                       "criteria": ["a one-liner", "routine work", "needs real reasoning"]},
    }})["states"][0]["answers"]["difficulty"]
    frontier = a["score"] >= 1.5
    saved += not frontier
    print(f"{'frontier' if frontier else 'local   '}  score {a['score']:.2f}  {text[:56]}")
print(f"\n{saved}/{len(REQUESTS)} handled locally")
