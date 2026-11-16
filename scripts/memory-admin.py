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

    rename = sub.add_parser("rename-user")
    rename.add_argument("--user", required=True)
    rename.add_argument("--name", required=True)

    delete_u = sub.add_parser("delete-user")
    delete_u.add_argument("--user", required=True)

    fact = sub.add_parser("set-fact")
    fact.add_argument("--user", required=True)
    fact.add_argument("--key", required=True)
    fact.add_argument("--value", required=True)
    fact.add_argument("--public", action="store_true")

    del_fact = sub.add_parser("delete-fact")
    del_fact.add_argument("--user", required=True)
    del_fact.add_argument("--key", required=True)

    clear_bio = sub.add_parser("clear-biometrics")
    clear_bio.add_argument("--user", required=True)
    clear_bio.add_argument("--modality", choices=["face", "voice", "all"], default="all")

    sub.add_parser("list")

    show = sub.add_parser("show")
    show.add_argument("--user", required=True)

    args = parser.parse_args()
    store = MemoryStore()
    if args.command == "list":
        print(json.dumps(store.list_profiles(), ensure_ascii=False, indent=2))
    elif args.command == "create-user":
        store.ensure_user(args.user, args.name)
        print(f"Created/updated {args.user}")
    elif args.command == "rename-user":
        store.rename_user(args.user, args.name)
        print(f"Renamed {args.user} -> {args.name}")
    elif args.command == "delete-user":
        removed = store.delete_user(args.user)
        print(f"Deleted {args.user}: {removed}")
    elif args.command == "set-fact":
        store.upsert_fact(
            args.user,
            args.key,
            args.value,
            visibility="public" if args.public else "private",
        )
        print(f"Saved {args.key}")
    elif args.command == "delete-fact":
        removed = store.delete_fact(args.user, args.key)
        print(f"Deleted {args.key}: {removed}")
    elif args.command == "clear-biometrics":
        modality = None if args.modality == "all" else args.modality
        removed = store.clear_biometrics(args.user, modality)
        print(f"Cleared {removed} templates for {args.user} ({args.modality})")
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
                    "face_templates": store.biometric_count(args.user, "face"),
                    "voice_templates": store.biometric_count(args.user, "voice"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    store.close()


if __name__ == "__main__":
    main()
