-- tests/assert_no_future_messages.sql
select *
from {{ ref('fct_messages') }}
where date_key > (
    select max(date_key) from {{ ref('dim_dates') }} where full_date <= current_date
)
