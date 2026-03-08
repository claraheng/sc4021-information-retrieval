import asyncio
import csv
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from playwright._impl._errors import TargetClosedError

QUERY = '"electric vehicles" lang:en filter:replies'
MAX_RECORDS = 400

STATE_DIR = Path(__file__).parent / ".pw_profile"
OUTPUT_DIR = Path(__file__).parent / "twitter_daily_runs"


async def is_logged_in(page) -> bool:
    try:
        await page.wait_for_selector('nav[role="navigation"]', timeout=8000)
        return True
    except PWTimeout:
        return False


async def ensure_logged_in(page):
    await page.goto("https://x.com/home", wait_until="load", timeout=60_000)

    if await is_logged_in(page):
        print("Detected: already logged in ✅")
        return

    print("Please log in manually in the Chrome window.")
    await page.goto("https://x.com/i/flow/login", wait_until="load", timeout=60_000)
    await page.wait_for_selector('nav[role="navigation"]', timeout=300_000)
    print("Login detected ✅")


def build_output_csv_path() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return OUTPUT_DIR / f"day1_electric_vehicles_{timestamp}.csv"


def save_results_to_csv(results, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "author", "text", "url", "query"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Autosaved {len(results)} rows -> {output_path.name}")


async def extract_visible_tweets(page, results, seen):
    tweets = await page.query_selector_all('article[data-testid="tweet"]')
    added = 0

    for t in tweets:
        if len(results) >= MAX_RECORDS:
            break

        a = await t.query_selector('a[href*="/status/"]')
        if not a:
            continue

        href = await a.get_attribute("href")
        if not href or "/status/" not in href:
            continue

        tid = href.split("/status/")[-1].split("?")[0]
        if not tid or tid in seen:
            continue

        text_el = await t.query_selector('div[data-testid="tweetText"]')
        text = (await text_el.inner_text()) if text_el else ""

        author_el = await t.query_selector('div[data-testid="User-Name"]')
        author = (await author_el.inner_text()) if author_el else ""
        author = author.split("\n")[0].strip() if author else ""

        results.append({
            "id": tid,
            "author": author,
            "text": text.strip(),
            "url": f"https://x.com{href}",
            "query": QUERY,
        })
        seen.add(tid)
        added += 1

    return added


async def main():
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(STATE_DIR),
            channel="chrome",
            headless=False,
        )
        page = context.pages[0] if context.pages else await context.new_page()

        output_csv = build_output_csv_path()
        results = []
        seen = set()

        try:
            await ensure_logged_in(page)

            search_url = "https://x.com/search?q=" + quote(QUERY, safe="") + "&src=typed_query&f=live"
            await page.goto(search_url, wait_until="load", timeout=60_000)
            await page.wait_for_selector('article[data-testid="tweet"]', timeout=60_000)

            no_new_rounds = 0

            while len(results) < MAX_RECORDS:
                if page.is_closed():
                    print("Page closed. Ending safely.")
                    break

                added = await extract_visible_tweets(page, results, seen)

                if added > 0:
                    no_new_rounds = 0
                    print(f"Collected {len(results)} / {MAX_RECORDS} (+{added})")
                    save_results_to_csv(results, output_csv)
                else:
                    no_new_rounds += 1
                    print(f"No new tweets this round. Total = {len(results)}")

                if len(results) >= MAX_RECORDS:
                    break

                await page.mouse.wheel(0, 2500)
                await page.wait_for_timeout(2000)

                if no_new_rounds >= 8:
                    print("No new tweets for several rounds. Ending this run.")
                    break

        except KeyboardInterrupt:
            print("Stopped by user.")
        except TargetClosedError:
            print("Page/context/browser closed unexpectedly.")
        finally:
            save_results_to_csv(results, output_csv)
            print(f"Final saved count: {len(results)}")
            await context.close()


if __name__ == "__main__":
    asyncio.run(main())