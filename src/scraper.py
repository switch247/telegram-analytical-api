# Telegram Scraper for Medical Business Channels
# Task 1: Data Scraping and Collection

import os
import json
from datetime import datetime
from pathlib import Path
import asyncio
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import MessageMediaPhoto
from telethon.tl.functions.messages import GetHistoryRequest

from src.config.constants import RAW_DIR
from src.utils import get_logger

# --- CONFIGURATION ---
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME = os.getenv("TELEGRAM_SESSION", "telegram_scraper")


# List of Telegram channels to scrape (no placeholders)
CHANNELS = [
	"https://t.me/CheMed123",
	"https://t.me/lobelia4cosmetics",
	"https://t.me/tikvahpharma",
	# For additional channels, see: https://et.tgstat.com/medicine
]

RAW_MESSAGES_DIR = RAW_DIR / "telegram_messages"
IMAGES_DIR = RAW_DIR / "images"
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


# DRY_RUN: If True, skip actual file operations but log as if successful
DRY_RUN =  os.getenv("DRY_RUN", "0") == "1"
logger = get_logger("telegram_scraper")

def sanitize_filename(name):
	return "".join(c if c.isalnum() or c in ("-_.") else "_" for c in name)


async def scrape_channel(client, channel_url, message_limit=100):
	channel_name = channel_url.split("/")[-1]
	channel_name_safe = sanitize_filename(channel_name)
	today = datetime.now().strftime("%Y-%m-%d")
	out_dir = RAW_MESSAGES_DIR / today
	if not DRY_RUN:
		out_dir.mkdir(parents=True, exist_ok=True)
		images_dir = IMAGES_DIR / channel_name_safe
		images_dir.mkdir(parents=True, exist_ok=True)
	else:
		images_dir = IMAGES_DIR / channel_name_safe
	json_path = out_dir / f"{channel_name_safe}.json"
	scraped_messages = []
	total = 0
	try:
		logger.info(f"Scraping channel: {channel_name} ({channel_url})")
		entity = await client.get_entity(channel_url)
		offset_id = 0
		limit = min(100, message_limit)  # Telegram API max is 100 per request
		while total < message_limit:
			fetch_limit = min(limit, message_limit - total)
			history = await client(GetHistoryRequest(
				peer=entity,
				offset_id=offset_id,
				offset_date=None,
				add_offset=0,
				limit=fetch_limit,
				max_id=0,
				min_id=0,
				hash=0
			))
			messages = history.messages
			if not messages:
				break
			for msg in messages:
				msg_dict = {
					"message_id": msg.id,
					"channel_name": channel_name,
					"message_date": msg.date.isoformat() if msg.date else None,
					"message_text": msg.message,
					"views": getattr(msg, "views", None),
					"forwards": getattr(msg, "forwards", None),
					"has_media": bool(msg.media),
					"image_path": None
				}
				# Download image if present
				if isinstance(msg.media, MessageMediaPhoto):
					img_path = images_dir / f"{msg.id}.jpg"
					if DRY_RUN:
						msg_dict["image_path"] = str(img_path)
						logger.info(f"Downloaded image for message {msg.id} in channel {channel_name} to {img_path}")
					else:
						try:
							await client.download_media(msg, file=img_path)
							msg_dict["image_path"] = str(img_path)
						except Exception as e:
							logger.error(f"Failed to download image for message {msg.id}: {e}")
				scraped_messages.append(msg_dict)
				total += 1
				if total >= message_limit:
					break
			if len(messages) < fetch_limit or total >= message_limit:
				break
			offset_id = messages[-1].id
		# Save messages to JSON or log as if saved
		if DRY_RUN:
			logger.info(f"Saved {total} messages for channel {channel_name} to {json_path}")
		else:
			with open(json_path, "w", encoding="utf-8") as f:
				json.dump(scraped_messages, f, ensure_ascii=False, indent=2)
		logger.info(f"Scraped {total} messages from {channel_name} (limit: {message_limit}).")
		logger.info(f"Saved to {json_path}")
	except FloodWaitError as e:
		logger.error(f"FloodWaitError: Must wait {e.seconds} seconds. Channel: {channel_name}")
	except Exception as e:
		logger.error(f"Error scraping {channel_name}: {e}")


async def main():
	logger.info("Starting Telegram scraping...")
	async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
		for channel in CHANNELS:
			await scrape_channel(client, channel, message_limit=500)
	logger.info("Scraping complete.")

if __name__ == "__main__":
	if not API_ID or not API_HASH:
		logger.error("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set as environment variables.")
	else:
		asyncio.run(main())
