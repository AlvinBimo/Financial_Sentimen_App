import scrapy
import dateparser
import re
from scrapy.http import Request
from urllib.parse import urljoin, quote_plus
from datetime import datetime
 
class KompasSpider(scrapy.Spider):
    name = "kompas"
    allowed_domains = ["kompas.com"]
 
    custom_settings = {
        'FEEDS': {
            'kompas_data.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'fields': None,
                'overwrite': True,
            },
        },
        'FEED_EXPORT_ENCODING': 'utf-8',
    }
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = kwargs.get('query', '')
        self.encoded_query = quote_plus(self.query)
        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')
        self.exact_match = kwargs.get('exact_match', True)
        self.stop_on_non_match = kwargs.get('stop_on_non_match', True)
        self.non_match_threshold = 5
        self.current_non_matches = 0
 
        if self.query:
            base = f"https://search.kompas.com/search?q={self.encoded_query}&site_id=all&type=article"
            self.start_urls = [base + (self.get_date_query() if self.start_date and self.end_date else "&last_date=all")]
        else:
            self.start_urls = []
 
    # ---------- utility ----------
    def get_date_query(self):
        return f"&start_date={self.start_date}&end_date={self.end_date}"
 
    def parse_kompas_date(self, raw: str) -> datetime | None:
        """Bersihkan prefix lalu parse dengan dateparser."""
        if not raw:
            return None
        cleaned = re.sub(r'^\s*Kompas\.com\s*[-,]?\s*', '', raw.strip(), flags=re.I)
        return dateparser.parse(
            cleaned,
            languages=['id'],
            settings={'TIMEZONE': 'Asia/Jakarta'}
        )
    # ------------------------------
 
    def parse(self, response):
        if 'search.kompas.com' not in response.url:
            return
 
        for article in response.css('div.articleItem'):
            url = article.css('a.article-link::attr(href)').get()
            if not url or 'video.kompas.com' in url:
                continue
            yield Request(
                url=url + '?page=all',
                callback=self.parse_article,
                meta={'title': article.css('h2.articleTitle::text').get('')}
            )
 
        if self.stop_on_non_match and self.current_non_matches >= self.non_match_threshold:
            self.logger.info(f"Stop after {self.current_non_matches} consecutive non-matches")
            return
 
        next_page = response.css('a.paging__link--next::attr(href)').get()
        if next_page:
            yield Request(url=urljoin(response.url, next_page), callback=self.parse)
 
    def parse_article(self, response):
        title = response.css('h1.read__title::text').get('')
        title = title.strip() or response.meta.get('title', '')
 
        body_parts = []
        for p in response.css('div.read__content div.clearfix p'):
            txt = p.xpath('string(.)').get('').strip()
            if txt and not any(txt.lower().startswith(w) for w in
                               ["baca juga:", "baca selengkapnya", "(baca:"]):
                body_parts.append(txt)
        body = ' '.join(body_parts)
 
        is_match = True
        if self.exact_match:
            is_match = (self.query.lower() in title.lower()) or (self.query.lower() in body.lower())
 
        if self.exact_match and not is_match:
            self.current_non_matches += 1
        else:
            self.current_non_matches = 0
 
        if is_match or not self.exact_match:
            date = self.parse_kompas_date(response.css('div.read__time::text').get(''))
            yield {
                'title': title,
                'link': response.url,
                'date': date.strftime('%Y-%m-%d %H:%M:%S') if date else '',
                'body': body,
                'source': 'kompas',
                'query': self.query
            }