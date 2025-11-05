import scrapy
from scrapy.http import Request
from urllib.parse import urljoin
from datetime import datetime
import locale
from urllib.parse import quote_plus

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
        super(KompasSpider, self).__init__(*args, **kwargs)

        self.query = kwargs.get('query', '')
        self.encoded_query = quote_plus(self.query)

        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')

        self.exact_match = kwargs.get('exact_match', True)
        self.stop_on_non_match = kwargs.get('stop_on_non_match', True)
        self.non_match_threshold = 5
        self.current_non_matches = 0

        if self.query:
            query_url = f"https://search.kompas.com/search?q={self.encoded_query}&site_id=all&type=article"
            if self.start_date and self.end_date:
                self.start_urls = [query_url + self.get_date_query()]
            else:
                self.start_urls = [query_url + "&last_date=all"]
        else:
            self.start_urls = []

        try:
            locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')
        except locale.Error:
            print("Indonesian locale not available. Trying with default locale.")
            pass

    def get_date_query(self):
        # start_date = datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        # end_date = datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        return f"&start_date={self.start_date}&end_date={self.end_date}"

    def parse(self, response):
        if 'search.kompas.com' in response.url:
            articles_list = response.css('div.articleItem')

            for article in articles_list:
                article_url = article.css('a.article-link::attr(href)').get()
                article_url += '?page=all'
                article_title = article.css('h2.articleTitle::text').get()

                if article_url and not 'video.kompas.com' in article_url:
                    yield Request(
                        url=article_url,
                        callback=self.parse_article,
                        meta={'title': article_title}
                    )

            if self.stop_on_non_match and self.current_non_matches >= self.non_match_threshold:
                self.logger.info(f"Stopping pagination after {self.current_non_matches} consecutive non-matches")
                return

            next_page = response.css('a.paging__link--next::attr(href)').get()
            if next_page:
                next_page_url = urljoin(response.url, next_page)
                yield Request(next_page_url, callback=self.parse)

    def parse_article(self, response):
        title = response.css('h1.read__title::text').get()
        title = title.strip() if title else response.meta.get('search_title', '')

        body = []
        for content in response.css('div.read__content div.clearfix p'):
            body_part = content.xpath('string(.)').get()
            if body_part:
                filter_words = ["baca juga:", "baca selengkapnya", "(baca:"]
                body_part = body_part.strip()
                if not any(body_part.strip().lower().startswith(w) for w in filter_words):
                    body.append(body_part)
        body = ' '.join(body)

        is_match = True
        if self.exact_match:
            is_match = (self.query.lower() in title.lower()) or (self.query.lower() in body.lower())

        self.logger.debug(f"query: {self.query}, title: {title}")

        if self.exact_match and not is_match:
            self.current_non_matches += 1
            self.logger.debug(f"{self.name} - Non-match #{self.current_non_matches}: {response.url}")
        else:
            self.current_non_matches = 0

        if is_match or not self.exact_match:
            date = response.css('div.read__time::text').get()
            if date:
                date = date.split('-')[1].strip()
                try:
                    date = datetime.strptime(date, "%d/%m/%Y, %H:%M WIB")
                except ValueError:
                    date = None

            yield {
                'title': title,
                'link': response.url,
                'date': date,
                'body': body,
                'source': 'kompas',
                'query': self.query
            }