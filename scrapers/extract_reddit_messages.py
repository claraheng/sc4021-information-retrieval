#!/usr/bin/env python3
"""Extract recent Reddit posts and comments into a schema-defined JSON file."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in missing dependency environments.
    yaml = None


USER_AGENT = "reddit-ev-extractor/1.0"
DEFAULT_SOURCE_TARGET = "https://www.reddit.com/r/electricvehicles/"
LOGGER = logging.getLogger("reddit_ev_extractor")


@dataclass(frozen=True)
class CandidateMessage:
    kind: str
    source_id: str
    text: str
    stars: int


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install dependencies from requirements.txt first.")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = {
        "source": {
            "subreddit": raw.get("source", {}).get("subreddit", "electricvehicles"),
        },
        "collection": {
            "target_messages": int(raw.get("collection", {}).get("target_messages", 2000)),
            "mode": raw.get("collection", {}).get("mode", "recent_backfill"),
        },
        "fetch": {
            "posts_page_size": int(raw.get("fetch", {}).get("posts_page_size", 100)),
            "max_posts_to_scan": int(raw.get("fetch", {}).get("max_posts_to_scan", 5000)),
            "request_delay_seconds": float(raw.get("fetch", {}).get("request_delay_seconds", 1.0)),
            "max_retries": int(raw.get("fetch", {}).get("max_retries", 5)),
            "retry_backoff_seconds": float(raw.get("fetch", {}).get("retry_backoff_seconds", 2.0)),
            "user_agent": raw.get("fetch", {}).get("user_agent", USER_AGENT),
            "comments_sort": raw.get("fetch", {}).get("comments_sort", "new"),
        },
        "keywords": {
            "ev_keywords": list(raw.get("keywords", {}).get("ev_keywords", [])),
            "other_keyword_lists": raw.get("keywords", {}).get("other_keyword_lists", {}),
        },
        "output": {
            "path": raw.get("output", {}).get("path", "reddit_ev_messages.json"),
        },
        "runtime": {
            "progress_log_every_posts": int(raw.get("runtime", {}).get("progress_log_every_posts", 25)),
            "checkpoint_path": raw.get("runtime", {}).get("checkpoint_path", "output/reddit_ev_messages.checkpoint.json"),
            "checkpoint_every_matches": int(raw.get("runtime", {}).get("checkpoint_every_matches", 50)),
            "resume_from_checkpoint": bool(raw.get("runtime", {}).get("resume_from_checkpoint", True)),
        },
    }

    ev_keywords = [keyword.strip() for keyword in config["keywords"]["ev_keywords"] if keyword.strip()]
    if not ev_keywords:
        raise ValueError("Config must define at least one entry in keywords.ev_keywords.")
    config["keywords"]["ev_keywords"] = ev_keywords

    if config["collection"]["mode"] != "recent_backfill":
        raise ValueError("Only collection.mode=recent_backfill is supported.")

    if config["collection"]["target_messages"] <= 0:
        raise ValueError("collection.target_messages must be greater than zero.")
    if config["fetch"]["posts_page_size"] <= 0 or config["fetch"]["posts_page_size"] > 100:
        raise ValueError("fetch.posts_page_size must be between 1 and 100.")
    if config["fetch"]["max_posts_to_scan"] <= 0:
        raise ValueError("fetch.max_posts_to_scan must be greater than zero.")
    if config["fetch"]["request_delay_seconds"] < 0:
        raise ValueError("fetch.request_delay_seconds must be non-negative.")
    if config["fetch"]["max_retries"] < 0:
        raise ValueError("fetch.max_retries must be non-negative.")
    if config["fetch"]["retry_backoff_seconds"] < 0:
        raise ValueError("fetch.retry_backoff_seconds must be non-negative.")
    if config["runtime"]["progress_log_every_posts"] <= 0:
        raise ValueError("runtime.progress_log_every_posts must be greater than zero.")
    if config["runtime"]["checkpoint_every_matches"] <= 0:
        raise ValueError("runtime.checkpoint_every_matches must be greater than zero.")

    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.target_messages is not None:
        if args.target_messages <= 0:
            raise ValueError("--target-messages must be greater than zero.")
        config["collection"]["target_messages"] = args.target_messages

    if args.ignore_keywords:
        config["keywords"]["ev_keywords"] = []

    if args.ignore_checkpoint:
        config["runtime"]["resume_from_checkpoint"] = False

    return config


def compile_keyword_patterns(keywords: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(re.escape(keyword), re.IGNORECASE) for keyword in keywords]


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return ""
    if normalized.lower() in {"[deleted]", "[removed]"}:
        return ""
    return normalized


def compose_post_text(title: str | None, body: str | None) -> str:
    clean_title = normalize_text(title)
    clean_body = normalize_text(body)

    if clean_title and clean_body:
        return f"{clean_title}\n\n{clean_body}"
    return clean_title or clean_body


def text_matches_ev_keywords(text: str, patterns: list[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    return any(pattern.search(text) for pattern in patterns)


def build_output_entry(candidate: CandidateMessage, source_target: str = DEFAULT_SOURCE_TARGET) -> dict[str, Any]:
    return {
        "doc_id": f"reddit:{candidate.kind}:{candidate.source_id}",
        "platform": "Reddit",
        "source_target": source_target,
        "text": candidate.text,
        "stars": candidate.stars,
    }


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def build_timestamped_output_path(path: Path, timestamp: str | None = None) -> Path:
    effective_timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{effective_timestamp}{path.suffix}")


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} is not a JSON object.")
    return payload


def write_checkpoint(
    path: Path,
    *,
    subreddit: str,
    after: str | None,
    scanned_posts: int,
    collected: list[dict[str, Any]],
) -> None:
    payload = {
        "subreddit": subreddit,
        "after": after,
        "scanned_posts": scanned_posts,
        "collected": collected,
    }
    save_json(path, payload)


def restore_checkpoint(
    checkpoint_payload: dict[str, Any],
    subreddit: str,
    source_target: str,
) -> tuple[list[dict[str, Any]], set[str], str | None, int]:
    if checkpoint_payload.get("subreddit") != subreddit:
        raise ValueError("Checkpoint subreddit does not match the configured subreddit.")

    collected = checkpoint_payload.get("collected", [])
    if not isinstance(collected, list):
        raise ValueError("Checkpoint collected payload must be a list.")

    restored: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in collected:
        if not isinstance(entry, dict):
            raise ValueError("Checkpoint contains a non-object collected entry.")
        doc_id = entry.get("doc_id")
        if not isinstance(doc_id, str):
            raise ValueError("Checkpoint entry is missing a string doc_id.")
        entry["source_target"] = source_target
        restored.append(entry)
        seen_ids.add(doc_id)

    after = checkpoint_payload.get("after")
    if after is not None and not isinstance(after, str):
        raise ValueError("Checkpoint after cursor must be a string or null.")

    scanned_posts = int(checkpoint_payload.get("scanned_posts", 0))
    return restored, seen_ids, after, scanned_posts


class RedditClient:
    def __init__(
        self,
        subreddit: str,
        user_agent: str,
        request_delay_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> None:
        self.subreddit = subreddit
        self.user_agent = user_agent
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._last_request_ts = 0.0

    def _get_json(self, url: str, query: dict[str, Any] | None = None) -> Any:
        if query:
            url = f"{url}?{urlencode(query)}"

        for attempt in range(self.max_retries + 1):
            now = time.monotonic()
            elapsed = now - self._last_request_ts
            if elapsed < self.request_delay_seconds:
                time.sleep(self.request_delay_seconds - elapsed)

            request = Request(
                url=url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )

            try:
                with urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                self._last_request_ts = time.monotonic()
                return payload
            except HTTPError as exc:  # pragma: no cover - requires live network failures.
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise RuntimeError(f"Reddit request failed with HTTP {exc.code} for {url}") from exc
            except URLError as exc:  # pragma: no cover - requires live network failures.
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Reddit request failed for {url}: {exc.reason}") from exc

            time.sleep(self.retry_backoff_seconds * (attempt + 1))

        raise RuntimeError(f"Reddit request failed after retries for {url}")

    def fetch_new_posts(self, limit: int, after: str | None = None) -> dict[str, Any]:
        query = {"limit": limit, "raw_json": 1}
        if after:
            query["after"] = after
        return self._get_json(
            f"https://www.reddit.com/r/{self.subreddit}/new.json",
            query=query,
        )

    def fetch_comments(self, permalink: str, sort: str) -> list[Any]:
        permalink_path = permalink.rstrip("/")
        return self._get_json(
            f"https://www.reddit.com{permalink_path}.json",
            query={"limit": 500, "sort": sort, "raw_json": 1},
        )


def iter_comment_bodies(children: list[dict[str, Any]]) -> list[CandidateMessage]:
    messages: list[CandidateMessage] = []
    stack = list(reversed(children))

    while stack:
        node = stack.pop()
        if node.get("kind") != "t1":
            continue

        data = node.get("data", {})
        text = normalize_text(data.get("body"))
        source_id = data.get("id")
        if text and source_id:
            messages.append(
                CandidateMessage(
                    kind="comment",
                    source_id=source_id,
                    text=text,
                    stars=int(data.get("score", 0)),
                )
            )

        replies = data.get("replies")
        if isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            stack.extend(reversed(reply_children))

    return messages


def extract_post_candidate(post_data: dict[str, Any]) -> CandidateMessage | None:
    text = compose_post_text(post_data.get("title"), post_data.get("selftext"))
    source_id = post_data.get("id")
    if not text or not source_id:
        return None

    return CandidateMessage(
        kind="post",
        source_id=source_id,
        text=text,
        stars=int(post_data.get("score", 0)),
    )


def collect_messages(config: dict[str, Any]) -> list[dict[str, Any]]:
    subreddit = config["source"]["subreddit"]
    target_messages = config["collection"]["target_messages"]
    fetch_cfg = config["fetch"]
    patterns = compile_keyword_patterns(config["keywords"]["ev_keywords"])
    source_target = f"https://www.reddit.com/r/{subreddit}/"

    client = RedditClient(
        subreddit=subreddit,
        user_agent=fetch_cfg["user_agent"],
        request_delay_seconds=fetch_cfg["request_delay_seconds"],
        max_retries=fetch_cfg["max_retries"],
        retry_backoff_seconds=fetch_cfg["retry_backoff_seconds"],
    )

    runtime_cfg = config["runtime"]
    checkpoint_path = Path(runtime_cfg["checkpoint_path"])
    collected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    after: str | None = None
    scanned_posts = 0

    if runtime_cfg["resume_from_checkpoint"]:
        checkpoint_payload = load_checkpoint(checkpoint_path)
        if checkpoint_payload is not None:
            collected, seen_ids, after, scanned_posts = restore_checkpoint(
                checkpoint_payload,
                subreddit=subreddit,
                source_target=source_target,
            )
            LOGGER.info(
                "Resuming from checkpoint %s with %s collected messages after scanning %s posts.",
                checkpoint_path,
                len(collected),
                scanned_posts,
            )

    while len(collected) < target_messages and scanned_posts < fetch_cfg["max_posts_to_scan"]:
        listing = client.fetch_new_posts(limit=fetch_cfg["posts_page_size"], after=after)
        children = listing.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            if child.get("kind") != "t3":
                continue

            post_data = child.get("data", {})
            scanned_posts += 1
            if scanned_posts > fetch_cfg["max_posts_to_scan"]:
                break
            if scanned_posts % runtime_cfg["progress_log_every_posts"] == 0:
                LOGGER.info(
                    "Scanned %s posts and collected %s/%s matching messages.",
                    scanned_posts,
                    len(collected),
                    target_messages,
                )

            candidates: list[CandidateMessage] = []
            post_candidate = extract_post_candidate(post_data)
            if post_candidate is not None:
                candidates.append(post_candidate)

            permalink = post_data.get("permalink")
            if permalink:
                comment_payload = client.fetch_comments(permalink=permalink, sort=fetch_cfg["comments_sort"])
                if len(comment_payload) >= 2:
                    comment_listing = comment_payload[1]
                    comment_children = comment_listing.get("data", {}).get("children", [])
                    candidates.extend(iter_comment_bodies(comment_children))

            for candidate in candidates:
                doc_id = f"reddit:{candidate.kind}:{candidate.source_id}"
                if doc_id in seen_ids:
                    continue
                if not text_matches_ev_keywords(candidate.text, patterns):
                    continue

                seen_ids.add(doc_id)
                collected.append(build_output_entry(candidate, source_target=source_target))
                if len(collected) % runtime_cfg["checkpoint_every_matches"] == 0:
                    write_checkpoint(
                        checkpoint_path,
                        subreddit=subreddit,
                        after=after,
                        scanned_posts=scanned_posts,
                        collected=collected,
                    )
                    LOGGER.info(
                        "Checkpoint saved to %s with %s/%s messages collected.",
                        checkpoint_path,
                        len(collected),
                        target_messages,
                    )
                if len(collected) >= target_messages:
                    write_checkpoint(
                        checkpoint_path,
                        subreddit=subreddit,
                        after=after,
                        scanned_posts=scanned_posts,
                        collected=collected,
                    )
                    return collected

        after = listing.get("data", {}).get("after")
        write_checkpoint(
            checkpoint_path,
            subreddit=subreddit,
            after=after,
            scanned_posts=scanned_posts,
            collected=collected,
        )
        if not after:
            break

    raise RuntimeError(
        "Unable to collect the requested number of messages. "
        f"Collected {len(collected)} of {target_messages} after scanning {scanned_posts} posts."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--target-messages",
        type=int,
        help="Override collection.target_messages from the YAML config.",
    )
    parser.add_argument(
        "--ignore-keywords",
        action="store_true",
        help="Disable EV keyword filtering and keep all non-empty posts/comments.",
    )
    parser.add_argument(
        "--ignore-checkpoint",
        action="store_true",
        help="Start a fresh run without resuming from the checkpoint file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv or sys.argv[1:])
    config = apply_cli_overrides(load_config(Path(args.config)), args)
    output_path = build_timestamped_output_path(Path(config["output"]["path"]))

    results = collect_messages(config)
    save_json(output_path, results)

    print(f"Wrote {len(results)} messages to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
