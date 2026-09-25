from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, r"D:\ReachyAI\app")

from reachy_ai.memory import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Reachy persistent memory")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-user")
    create.add_argument("--user", required=True)
    create.add_argument("--name", required=True)

    fact = sub.add_parser("set-fact")
    fact.add_argument("--user", required=True)
    fact.add_argument("--key", required=True)
    fact.add_argument("--value", required=True)
    fact.add_argument("--public", action="store_true")

    show = sub.add_parser("show")
    show.add_argument("--user", required=True)

    args = parser.parse_args()
    store = MemoryStore()
    if args.command == "create-user":
        store.ensure_user(args.user, args.name)
        print(f"Created/updated {args.user}")
    elif args.command == "set-fact":
        store.upsert_fact(
            args.user,
            args.key,
            args.value,
            visibility="public" if args.public else "private",
        )
        print(f"Saved {args.key}")
    elif args.command == "show":
        print(
            json.dumps(
                {
                    "user": args.user,
                    "name": store.user_name(args.user),
                    "facts": store.facts(args.user),
                    "episodes": store.recent_episodes(args.user, 10),
                    "daily_english": (
                        store.get_daily_english(args.user).__dict__
                        if store.get_daily_english(args.user)
                        else None
                    ),
                    "face_templates": len(store.biometric_templates("face")),
                    "voice_templates": len(store.biometric_templates("voice")),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    store.close()


if __name__ == "__main__":
    main()
