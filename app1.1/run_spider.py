import os
import sys
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import pandas as pd
from pathlib import Path
from typing import List, Optional

scrapy_project_path = os.path.join(os.path.dirname(__file__), 'web_scraping')
sys.path.append(scrapy_project_path)

os.environ['SCRAPY_SETTINGS_MODULE'] = 'web_scraping.settings'


def run_spider(
        output_file: Path,
        max_items: Optional[int] = None,
        spiders: List[str] = [],
        **kwargs
) -> None:
    """
    Run Scrapy spiders, merge their outputs, and save to a combined CSV.
    """
    settings = get_project_settings()
    if max_items:
        settings.set('CLOSESPIDER_ITEMCOUNT', max_items)

    process = CrawlerProcess(settings)

    for spider in spiders:
        process.crawl(spider, **kwargs)

    process.start()

    dfs = []
    output_files = [f"{spider}_data.csv" for spider in spiders]
    for file in output_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Read {len(df)} records from {file}")
        else:
            print(f"Warning: {file} not found!")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Merged data saved to {output_file} ({len(combined_df)} records)")

        for file in output_files:
            try:
                os.remove(file)
                print(f"🗑️ Deleted temporary file: {file}")
            except OSError as e:
                print(f"⚠️ Error deleting {file}: {e}")
    else:
        print("❌ No data to merge!")
