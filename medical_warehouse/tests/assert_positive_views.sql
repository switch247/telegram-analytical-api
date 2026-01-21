-- tests/assert_positive_views.sql
select *
from {{ ref('fct_messages') }}
where view_count < 0
