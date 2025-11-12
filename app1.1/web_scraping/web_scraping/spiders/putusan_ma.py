import scrapy
from urllib.parse import quote_plus, urljoin
from datetime import datetime
from scrapy.http import HtmlResponse

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

        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')

        if self.start_date:
            self.start_date = datetime.strptime(self.start_date, '%Y-%m-%d').date()
        if self.end_date:
            self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d').date()

        if self.query:
            self.start_urls = [f'https://putusan3.mahkamahagung.go.id/search.html?q={self.encoded_query}&page=1']
        else:
            self.start_urls = []

    def parse(self, response):
        results = response.css('div.entry-c')

        print("INI HASIL: ", results)
        print("INI JUGA HASIL: ",self.start_date, self.end_date)
        for r in results:
            title = r.css('strong a::text').get('').strip()
            link = r.css('strong a::attr(href)').get()
            small = r.css('div.small')
            tanggal_putus = None

            for s in small.css('strong'):
                label = s.xpath('normalize-space(text())').get()
                next_text = s.xpath('following-sibling::text()[1]').get()
                if label and 'Putus' in label and next_text:
                    tanggal_putus = next_text.strip().replace('—', '').strip()
                    break
            
            print(f"DEBUG: title={title}, link={link}, tanggal_putus={tanggal_putus}")

            if tanggal_putus:
                try:
                    tanggal_obj = datetime.strptime(tanggal_putus, '%d-%m-%Y').date()
                except ValueError:
                    tanggal_obj = None

                if tanggal_obj:
                    if self.start_date and tanggal_obj < self.start_date:
                        continue
                    if self.end_date and tanggal_obj > self.end_date:
                        continue

            if title and link and tanggal_putus:
                hasil = {
                    'title': title,
                    'link': link,
                    'tanggal_putus': tanggal_putus,
                    'query': self.query
                }
                yield hasil
            
            with open('hasil_putusan_ma.txt', 'w', encoding='utf-8') as f:
                f.write(f"{title}\n{link}\n{tanggal_putus}\n\n")

        for page in range(2, 2):
            next_page = f'https://putusan3.mahkamahagung.go.id/search.html?q={self.encoded_query}&page={page}'
            yield scrapy.Request(next_page, callback=self.parse)

        # next_page = response.css('a.page-link[rel="next"]::attr(href)').get()
        # if next_page:
        #     next_page_url = urljoin(response.url, next_page)
        #     yield scrapy.Request(url=next_page_url, callback=self.parse)
