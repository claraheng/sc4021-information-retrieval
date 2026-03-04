import asyncio
from pathlib import Path
from urllib.parse import quote

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

QUERY = '("electric vehicles" OR "electric vehicle" OR "electric car" OR "electric cars" OR "EV charging" OR "battery electric") lang:en -filter:replies'
MAX_RECORDS = 10

STATE_DIR = Path(__file__).parent / ".pw_profile"  # persistent browser profile dir


async def is_logged_in(page) -> bool:
    try:
        await page.wait_for_selector('nav[role="navigation"]', timeout=8000)
        return True
    except PWTimeout:
        return False


async def ensure_logged_in(page):
    # Prefer landing somewhere stable
    await page.goto("https://x.com/home", wait_until="load", timeout=60_000)

    if await is_logged_in(page):
        print("Detected: already logged in ✅")
        return

    print("\nNot logged in.")
    print("Please log in manually in the Chrome window.\n")

    # Go to login flow only if needed
    await page.goto("https://x.com/i/flow/login", wait_until="load", timeout=60_000)
    await page.wait_for_selector('nav[role="navigation"]', timeout=300_000)
    print("Login detected ✅")


async def main():
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(STATE_DIR),
            channel="chrome",
            headless=False,
        )
        page = await context.new_page()

        try:
            await ensure_logged_in(page)

            search_url = "https://x.com/search?q=" + quote(QUERY, safe="") + "&src=typed_query&f=live"
            await page.goto(search_url, wait_until="load", timeout=60_000)

            await page.wait_for_selector('article[data-testid="tweet"]', timeout=60_000)

            results = []
            seen = set()

            while len(results) < MAX_RECORDS:
                tweets = await page.query_selector_all('article[data-testid="tweet"]')

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
                    author = author.split("\n")[0].strip() if author else None

                    results.append({
                        "id": tid,
                        "author": author,
                        "text": text.strip(),
                        "url": f"https://x.com{href}",
                    })
                    seen.add(tid)

                await page.mouse.wheel(0, 2500)
                await page.wait_for_timeout(1500)

            print("\n===== TEST OUTPUT (10 Tweets) =====\n")
            for i, t in enumerate(results, 1):
                print(f"{i}. {t['author']}")
                print(f"   ID: {t['id']}")
                print(f"   URL: {t['url']}")
                print(f"   Text: {t['text'][:200]}")
                print("-" * 60)

        except Exception as e:
            # helpful debug artifact
            try:
                await page.screenshot(path="pw_error.png", full_page=True)
                print("Saved screenshot: pw_error.png")
            except Exception:
                pass
            raise
        finally:
            await context.close()


if __name__ == "__main__":
    asyncio.run(main())