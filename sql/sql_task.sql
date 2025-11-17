-- 1.Write a query that denotes the time difference for any given timezone and UTC (i.e. MST is -7).
-- a. You can ignore any rows with missing Timezone values.

select 
    *,
    -1 * case 
        when timezone is null then null

        when lower(timezone) like '%central%' 
            then datediff(hour, getutcdate(), getutcdate() at time zone 'central standard time')

        when lower(timezone) like '%mountain%' 
            then datediff(hour, getutcdate(), getutcdate() at time zone 'mountain standard time')

        when lower(timezone) like '%eastern%' 
            then datediff(hour, getutcdate(), getutcdate() at time zone 'eastern standard time')

        when lower(timezone) like '%pacific%' 
            then datediff(hour, getutcdate(), getutcdate() at time zone 'pacific standard time')

        else null
    end as utc_offset_hours
from us_traffic_accidents
where timezone is not null;


-- 2. Write queries related to the Temperature_Range(F) column:

--      i. Convert the temperatures into Celsius for your stakeholders. Use the
--         average from the range.
--              1. to get C from F, use the following formula
--                 C = (F − 32) * 5/9

--      ii. Calculate the median temperature (celsius) for all the accidents in a new
--          query.
--      iii. Impute the missing data

with celsius as (
    select
        (
            (min_temperature_f + max_temperature_f) / 2.0 - 32
        ) * 5.0/9.0 as avg_temperature_c
    from us_traffic_accidents
    where 
        min_temperature_f is not null and 
        max_temperature_f is not null
)

select
    top 1 percentile_cont(0.5) within group (order by avg_temperature_c) over () as median_temperature_c
from celsius;

-- 3. Rank the Top 10 Cities by Total Road Length Impacted
--      a. Calculate the total road length impacted by accidents for each city.
--      b. Rank cities based on their total road length impacted. Include a column identifying
--         rank 1-10.
--      c. Return only the top 10 cities, along with their total road length and rank.

with city_totals as (
    select 
        city,
        sum(distancemi) as total_road_length
    from us_traffic_accidents
    where city is not null
    group by city
),

ranked as (
    select
        city,
        total_road_length,
        rank() over (order by total_road_length desc) as city_rank
    from city_totals
)

select
    city,
    total_road_length,
    city_rank
from ranked
where city_rank <= 10
order by city_rank;


