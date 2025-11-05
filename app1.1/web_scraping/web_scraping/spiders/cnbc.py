import scrapy
from scrapy.http import Request
from urllib.parse import urljoin
from datetime import datetime
import dateparser
from urllib.parse import quote_plus

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
    }

    def __init__(self, *args, **kwargs):
        super(CnbcSpider, self).__init__(*args, **kwargs)

        self.query = kwargs.get('query', '')
        self.encoded_query = quote_plus(self.query)
        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')
        self.exact_match = kwargs.get('exact_match', True)

        if self.query:
            query_url = f"https://www.cnbcindonesia.com/search?query={self.encoded_query}&tipe=artikel"
            print("QUERY URL: ", query_url)
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
                        meta={
                            'search_title': article_title.strip() if article_title else ''
                        }
                    )

            next_page = response.xpath('//a[@dtr-act="halaman selanjutnya"]/@href').get()
            if next_page:
                next_page_url = urljoin(response.url, next_page)
                yield Request(next_page_url, callback=self.parse)

    def parse_article(self, response):
        title = response.css('h1.mb-4::text').get().strip()
        title = title.strip() if title else response.meta.get('search_title', '')

        date = response.css('div.detail-head>div.text-gray::text').get()
        date = dateparser.parse(date.strip(), ["%d %B %Y %H:%M"], ['en', 'id']) if date else None

        # TODO: some (older) pages dont use this path, just text in 'div.detail-text div.detail-text'
        #       simply taking string(.) from that would be very unclean
        #       func like inner text in JS might work? any scrapy equivalent?
        #       for now this problem is simply ignored
        body = []
        for content in response.css('div.detail-text div.detail-text p'):
            body_part = content.xpath('string(.)').get()
            if body_part:
                filter_words = ["[gambas"]
                body_part = body_part.strip()
                if not any(body_part.strip().lower().startswith(w) for w in filter_words):
                    body.append(body_part)
        body = ' '.join(body)

        is_match = (self.query.lower() in title.lower()) or (self.query.lower() in body.lower()) or not self.exact_match

        if not (title and date and body) or not is_match:
            return

        yield {
            'title': title,
            'link': response.url,
            'date': date,
            'body': body,
            'source': 'cnbc',
            'query': self.query
        }