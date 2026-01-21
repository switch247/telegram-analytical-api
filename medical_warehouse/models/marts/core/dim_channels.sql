-- dim_channels.sql
with channel_stats as (
    select
        channel_name,
        channel_type,
        min(post_date) as first_post_date,
        max(post_date) as last_post_date,
        count(*) as total_posts,
        avg(view_count) as avg_views
    from {{ ref('stg_telegram_messages') }}
    group by channel_name, channel_type
)
select
    row_number() over (order by channel_name) as channel_key,
    channel_name,
    channel_type,
    first_post_date,
    last_post_date,
    total_posts,
    avg_views
from channel_stats
