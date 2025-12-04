import scrapy
from scrapy.http import Request
from urllib.parse import urljoin, quote_plus
from datetime import datetime
import dateparser
import re
 
class CnbcSpider(scrapy.Spider):
    name = "cnbc"
    allowed_domains = ["cnbcindonesia.com"]
 
    custom_settings = {
        'FEEDS': {
            'cnbc_data.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'fields': None,
                'overwrite': True,
            },
        },
        'FEED_EXPORT_ENCODING': 'utf-8',
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    }
 
    def __init__(self, *args, **kwargs):
        super(CnbcSpider, self).__init__(*args, **kwargs)
        self.query = kwargs.get('query', '')
        self.encoded_query = quote_plus(self.query)
        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')
        self.exact_match = kwargs.get('exact_match', False)  # default False
 
        if self.query:
            query_url = f"https://www.cnbcindonesia.com/search?query={self.encoded_query}&tipe=artikel"
            if self.start_date and self.end_date:
                self.start_urls = [query_url + self.get_date_query()]
            else:
                self.start_urls = [query_url]
        else:
            self.start_urls = []
 
    def get_date_query(self):
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%Y/%m/%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%Y/%m/%d")
        return f"&fromdate={start_date}&todate={end_date}"
 
    def _exact_sentence_match(self, text: str, query: str) -> bool:
        """
        True jika ada kalimat di `text` yang memuat seluruh kata `query`
        secara berurutan (case-insensitive).
        """
        if not query:
            return True
        pattern = r'\b{}\b'.format(re.escape(query.lower()))
        for sent in re.split(r'[.!?]', text.lower()):
            if re.search(pattern, sent):
                return True
        return False
 
    def parse(self, response):
        if '/search' in response.url:
            articles_list = response.css('article a')
            for article in articles_list:
                article_url = article.css('::attr(href)').get()
                article_title = article.css('h2::text').get()
                if article_url:
                    yield Request(
                        url=article_url,
                        callback=self.parse_article,
                        meta={'search_title': article_title.strip() if article_title else ''}
                    )
 
            next_page = response.xpath('//a[contains(text(),"Selanjutnya") or contains(@dtr-act,"halaman selanjutnya")]/@href').get()
            if next_page:
                yield Request(url=urljoin(response.url, next_page), callback=self.parse)
 
    def parse_article(self, response):
        title = response.css('h1.detail__title::text').get('')
        title = title.strip() or response.meta.get('search_title', '')
 
        date_raw = response.css('div.detail__date::text, span.date::text').get()
        date = dateparser.parse(date_raw, languages=['id']) if date_raw else None
        date = date or datetime.now()
 
        body_parts = []
        for sel in ['div.detail__body p', 'div.detail-text p', 'article p']:
            for p in response.css(sel):
                txt = p.xpath('string(.)').get('').strip()
                if txt and not txt.lower().startswith('[gambas'):
                    body_parts.append(txt)
        body = ' '.join(body_parts)
 
        full_text = f"{title} {body}"
        if not self._exact_sentence_match(full_text, self.query):
            return
 
        yield {
            'title': title,
            'link': response.url,
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'body': body,
            'source': 'cnbc',
            'query': self.query
        }