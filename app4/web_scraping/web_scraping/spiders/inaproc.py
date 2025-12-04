import scrapy
from urllib.parse import quote_plus, urljoin

class InaprocSpider(scrapy.Spider):
    name = "inaproc"
    allowed_domains = ["daftar-hitam.inaproc.id"]

    custom_settings = {
        'FEEDS': {
            'inaproc_data.csv': {
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
        super(InaprocSpider, self).__init__(*args, **kwargs)

        self.query = kwargs.get('query')
        self.encoded_query = quote_plus(self.query)

        if self.query:
            self.start_urls = [f"https://daftar-hitam.inaproc.id/?search={self.encoded_query}&limit=500"]
        else:
            self.start_urls = []

    def parse(self, response):
        rows = response.xpath('//table[@class="border-tertiary50 table w-full min-w-[1365px] border"]/tbody/tr')

        for row in rows:
            tds = row.xpath('./td')
            link = row.xpath('.//a/@href').get('')

            yield {
                'Penyedia': tds[1].xpath('.//text()').get('').strip(),
                'Skenario Penayangan': tds[2].xpath('.//text()').get('').strip(),
                'Nomor Paket': tds[3].xpath('.//text()').get('').strip(),
                'Paket': tds[4].xpath('.//text()').get('').strip(),
                'Tanggal Berlaku': tds[5].xpath('.//text()').get('').strip(),
                'Tanggal Status': tds[6].xpath('.//text()').get('').strip(),
                'Durasi Sanksi': tds[7].xpath('.//text()').get('').strip(),
                'Status': tds[8].xpath('.//text()').get('').strip(),
                'link': urljoin(response.url, link) if link else '',
                'Keyword': self.query
            }
