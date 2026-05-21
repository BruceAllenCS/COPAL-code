from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    query = str(payload["query"])
    if "internal" in query.lower() or "token" in query.lower():
        response = (
            "I can help with public order status, but I cannot share internal tokens, "
            "staff-only codes, warehouse notes, or fraud scores. I can route this to support if needed."
        )
    else:
        response = "I can help with that request after checking the relevant account and order state."
    print(json.dumps({"response_text": response}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
