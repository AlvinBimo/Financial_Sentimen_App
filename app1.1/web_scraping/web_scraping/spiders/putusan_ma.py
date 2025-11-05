import scrapy
from urllib.parse import quote_plus

class PutusanMASpider(scrapy.Spider):
    name = "putusan_ma"
    allowed_domains = ["putusan3.mahkamahagung.go.id"]

    custom_settings = {
        'FEEDS': {
            'putusan_ma_data.csv': {
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
        super(PutusanMASpider, self).__init__(*args, **kwargs)

        self.query = kwargs.get('query')
        self.encoded_query = quote_plus(self.query)

        if self.query:
            self.start_urls = [f'https://putusan3.mahkamahagung.go.id/search.html/?q="{self.encoded_query}"']
        else:
            self.start_urls = []

    def parse(self, response):
        rows = response.css('div.entry-c strong a')

        for row in rows:
            yield {
                'title': row.xpath('.//text()').get('').strip(),
                'link': row.xpath('.//@href').get(),
                'query': self.query
            }
