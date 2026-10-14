# Recorded API fixtures

Real responses recorded live on 2026-10-14. Tests must build `Window` objects that match these date ranges
(e.g. `Window(date(2026, 9, 14), date(2026, 10, 13))`). Do not re-record; tests must stay offline.

| File | Request | Content |
|---|---|---|
| `geocoding_rome.json` | `https://geocoding-api.open-meteo.com/v1/search?name=Rome&count=1&language=en&format=json` |  |
| `geocoding_milan.json` | `https://geocoding-api.open-meteo.com/v1/search?name=Milan&count=1&language=en&format=json` |  |
| `geocoding_not_found.json` | `https://geocoding-api.open-meteo.com/v1/search?name=Xqzvbnmlkj&count=1&language=en&format=json` |  |
| `air_quality_milan.json` | `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=45.46427&longitude=9.18951&hourly=pm2_5%2Cpm10&past_days=15&forecast_days=3&timezone=Europe%2FRome` | hourly 2026-09-29T00:00..2026-10-16T23:00 (432 h); pm10 nulls: 0 |
| `wiki_opensearch_ice_cream.json` | `https://en.wikipedia.org/w/api.php?action=opensearch&search=ice+cream&limit=1&namespace=0&format=json` |  |
| `wiki_opensearch_not_found.json` | `https://en.wikipedia.org/w/api.php?action=opensearch&search=xqzvbnmlkjhgf&limit=1&namespace=0&format=json` |  |
| `wiki_pageviews_ice_cream.json` | `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Ice_cream/daily/20260914/20261013` | 30 days, 2026091400..2026101300 |
| `fx_eur_usd.json` | `https://api.frankfurter.dev/v1/2026-09-14..2026-10-13?base=EUR&symbols=USD` | 22 business days, 2026-09-14..2026-10-13 |
| `energy_it.json` | `https://api.energy-charts.info/public_power?country=it&start=2026-10-10&end=2026-10-12` | 288 points (15 min), 2026-10-09 22:00Z..2026-10-12 21:45Z; trimmed to types ['Wind offshore', 'Wind onshore', 'Solar', 'Load'] |
| `weather_rome_history.json` | `https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=41.89193&longitude=12.51133&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max,sunshine_duration&start_date=2026-09-14&end_date=2026-10-13&timezone=Europe/Rome` | 30 days 2026-09-14..2026-10-13, no nulls |
| `weather_rome_forecast.json` | `https://api.open-meteo.com/v1/forecast?latitude=41.89193&longitude=12.51133&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max,sunshine_duration&forecast_days=7&timezone=Europe/Rome` | 7 days 2026-10-14..2026-10-20 (forecast) |
