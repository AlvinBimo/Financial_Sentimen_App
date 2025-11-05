import scrapy
from scrapy.http import Request
from urllib.parse import urljoin
from datetime import datetime
import dateparser
from urllib.parse import quote_plus

# TODO: find every supported urls to avoid unnecessary scraping
# supported_urls = [
#     'regional.kontan',
#     'lifestyle.kontan',
#     'pressrelease.kontan'
# ]

class KontanSpider(scrapy.Spider):
    name = "kontan"
    allowed_domains = ["kontan.co.id"]

    custom_settings = {
        'FEEDS': {
            'kontan_data.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'fields': None,
                'overwrite': True,
            },
        },
        'FEED_EXPORT_ENCODING': 'utf-8',
        'ROBOTSTXT_OBEY': False, # Searching won't work without this, sadly
    }

    def __init__(self, *args, **kwargs):
        super(KontanSpider, self).__init__(*args, **kwargs)

        self.query = kwargs.get('query', '')
        self.encoded_query = quote_plus(self.query)

        self.start_date = kwargs.get('start_date', '')
        self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d") if self.start_date else None

        self.end_date = kwargs.get('end_date', '')
        self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59) if self.end_date else None

        self.exact_match = kwargs.get('exact_match', True)

        self.start_urls = []

        if self.encoded_query:
            self.start_urls = [f"https://www.kontan.co.id/search/?search={self.encoded_query}"]

    def parse(self, response):
        if 'kontan.co.id/search' in response.url:
            articles_list = response.css('div.ket')
            next_page = response.xpath('//a[text()="Next »"]/@href').get()

            if articles_list:
                # first check if we're later than end_date range
                # check earliest date
                # if > end date, paginate
                earliest_date = articles_list[-1].css('span.font-gray::text').get()
                earliest_date = dateparser.parse(earliest_date, languages=['id']) if earliest_date else None
                if self.end_date and earliest_date and earliest_date > self.end_date:
                    yield Request(
                        urljoin(response.url, next_page),
                        callback=self.parse
                    )

                for article in articles_list:
                    article_url = article.css('h1 a::attr(href)').get()
                    article_title = article.css('h1 a::text').get()
                    article_date = article.css('span.font-gray::text').get()
                    article_date = dateparser.parse(article_date, languages=['id']) if article_date else None

                    # second check if we're earlier than start_date range
                    # check latest date
                    # if < start date, end pagination
                    if self.start_date and article_date < self.start_date:
                        return

                    if article_url:
                        yield Request(
                            url=urljoin(response.url, article_url),
                            callback=self.parse_article,
                            meta={
                                'search_title': article_title.strip() if article_title else '',
                                'search_date': article_date,
                            }
                        )

                    if next_page:
                        next_page_url = urljoin(response.url, next_page)
                        yield Request(next_page_url, callback=self.parse)

    def parse_article(self, response):
        title = response.meta.get('search_title', '')
        date = response.meta.get('search_date', '')

        if date and date.tzinfo:
            date = date.replace(tzinfo=None, microsecond=0)

        body_selector_path = ''
        if any(u in response.url for u in ['regional.kontan', 'lifestyle.kontan'] ):
            body_selector_path = 'div.ctn p'
        elif 'pressrelease.kontan' in response.url:
            body_selector_path = '#release-content p'
        else:
            body_selector_path = 'div.tmpt-desk-kon p'

        body = []
        for content in response.css(body_selector_path):
            body_part = content.xpath('string(.)').get()
            filter_words = ["baca juga", "selanjutnya", "menarik dibaca", "cek berita dan artikel", "simak juga", "lihat juga"]
            body_part = body_part.strip()
            if not any(body_part.strip().lower().startswith(w) for w in filter_words):
                body.append(body_part)
        body = ' '.join(body)

        is_match = (self.query.lower() in title.lower()) or (self.query.lower() in body.lower()) or not self.exact_match

        if not (title and date and body) or not is_match:
            return

        if body:
            yield {
                'title': title,
                'link': response.url,
                'date': date,
                'body': body,
                'source': 'kontan',
                'query': self.query
            }