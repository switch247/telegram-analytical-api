from pydantic import BaseModel
from typing import List, Optional

class TopProduct(BaseModel):
    product: str
    count: int

class ChannelActivity(BaseModel):
    channel_name: str
    total_posts: int
    avg_views: float
    first_post_date: str
    last_post_date: str

class MessageSearchResult(BaseModel):
    message_id: str
    channel_name: str
    message_text: str
    post_date: str
    view_count: int
    forward_count: int
    has_image: bool

class VisualContentStats(BaseModel):
    channel_name: str
    promotional_count: int
    product_display_count: int
    lifestyle_count: int
    other_count: int
