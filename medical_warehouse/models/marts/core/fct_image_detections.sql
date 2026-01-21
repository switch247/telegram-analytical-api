-- fct_image_detections.sql
with detections as (
    select
        channel_name,
        message_id,
        detected_class,
        confidence_score,
        image_category
    from processed.yolo_detections
)

select
    m.message_id,
    m.channel_key,
    m.date_key,
    d.detected_class,
    d.confidence_score,
    d.image_category
from detections d
left join {{ ref('fct_messages') }} m
    on d.message_id = m.message_id
