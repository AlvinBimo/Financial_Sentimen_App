import scrapy
from scrapy_playwright.page import PageMethod
from urllib.parse import urljoin, quote_plus
from datetime import datetime
import re
import dateparser


class CnnSpider(scrapy.Spider):
    name = "cnn"
    allowed_domains = ["cnnindonesia.com"]

    custom_settings = {
        'FEEDS': {
            'cnn_data.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'fields': None,
                'overwrite': True,
            },
        },
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'TWISTED_REACTOR': "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        'PLAYWRIGHT_BROWSER_TYPE': "chromium",
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            "headless": True,
        },
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 20000,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raw query for matching in title/body
        raw_query = kwargs.get('query') or ''
        self.query = raw_query.strip()

        # Encoded query for URL
        self.encoded_query = quote_plus(self.query) if self.query else ''

        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')

        # exact_match may come as string from command line
        self.exact_match = kwargs.get('exact_match', True)
        if isinstance(self.exact_match, str):
            self.exact_match = self.exact_match.lower() in ["true", "1", "yes", "y"]

        self.current_page = 1

    def start_requests(self):
        if not self.encoded_query:
            self.logger.error("No query provided!")
            return

        base_url = f"https://www.cnnindonesia.com/search?query={self.encoded_query}"

        if self.start_date and self.end_date:
            def format_date(date_str):
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                # cnn uses dd/mm/YYYY encoded as dd%2Fmm%2FYYYY
                return dt.strftime("%d%%2F%m%%2F%Y")

            date_query = f"&fromdate={format_date(self.start_date)}&todate={format_date(self.end_date)}"
            base_url += date_query

        yield scrapy.Request(
            base_url,
            self.parse_search_results,
            meta={
                'playwright': True,
                'playwright_page_methods': [
                    PageMethod('wait_for_selector', 'div.flex.flex-col.gap-5.nhl-list', timeout=10000),
                ],
                'playwright_include_page': False,
            }
        )

    def parse_search_results(self, response):
        articles = response.xpath(
            '//div[@class="flex flex-col gap-5 nhl-list"]//article/a/@href'
        ).getall()

        if not articles:
            self.logger.info("No more articles found. Stopping pagination.")
            return

        for article_url in articles:
            yield scrapy.Request(
                urljoin(response.url, article_url),
                self.parse_article,
                meta={
                    'playwright': False,  # not using playwright for article for now
                    'playwright_page_methods': [
                        PageMethod(
                            'wait_for_selector',
                            'h1.mb-2.text-\\[28px\\].leading-9.text-cnn_black',
                            timeout=10000
                        ),
                        PageMethod(
                            'wait_for_selector',
                            'div.text-cnn_grey.text-sm.mb-4',
                            timeout=10000
                        ),
                        PageMethod(
                            'wait_for_selector',
                            'div.detail-wrap p',
                            timeout=10000
                        ),
                    ],
                }
            )

        # Pagination
        self.current_page += 1
        base_without_page = response.url.split('&page=')[0]
        next_page = f"{base_without_page}&page={self.current_page}"

        yield scrapy.Request(
            next_page,
            self.parse_search_results,
            meta={
                'playwright': True,
                'playwright_page_methods': [
                    PageMethod('wait_for_selector', 'div.flex.flex-col.gap-5.nhl-list', timeout=10000),
                ],
            }
        )

    def parse_article(self, response):
        # Title
        title = response.xpath(
            '//h1[@class="mb-2 text-[28px] leading-9 text-cnn_black"]/text()'
        ).get()
        if title:
            title = title.strip()

        # Date
        date_text = response.xpath(
            '//div[@class="text-cnn_grey text-sm mb-4"]/text()'
        ).get()
        date = None
        if date_text:
            date_text = date_text.strip()
            date = dateparser.parse(date_text, languages=['id'])
            if date and date.tzinfo:
                date = date.replace(tzinfo=None)

        # Body
        body = ' '.join(
            response.xpath(
                '//div[contains(@class, "detail-wrap")]//p//text()'
            ).getall()
        ).strip()

        # If title and body both None/empty, nothing to do
        if not title and not body:
            return

        # Safe lowercase checks
        query_lower = (self.query or '').lower()
        title_lower = (title or '').lower()
        body_lower = (body or '').lower()

        if self.exact_match:
            # If query is empty, treat as match-all
            if query_lower:
                is_match = (query_lower in title_lower) or (query_lower in body_lower)
            else:
                is_match = True
        else:
            is_match = True

        # Skip if essential fields missing or query does not match
        if not (title and date and body and is_match):
            return

        yield {
            'title': title,
            'link': response.url,
            'date': date,
            'body': body,
            'source': 'cnn',
            'query': self.query,  # store raw query here
        }