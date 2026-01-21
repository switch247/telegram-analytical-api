with source as (
    select
        message_id,
        channel_name,
        channel_type,
        message_text,
        cast(post_date as timestamp) as post_date,
        cast(view_count as integer) as view_count,
        cast(forward_count as integer) as forward_count,
        image_url
    from {{ source('raw', 'telegram_messages') }}
),

cleaned as (
    select
        message_id,
        channel_name,
        channel_type,
        message_text,
        post_date,
        view_count,
        forward_count,
        image_url,
        length(message_text) as message_length,
        case when image_url is not null and image_url != '' then true else false end as has_image
    from source
    where message_text is not null and message_text != ''
)

select * from cleaned
